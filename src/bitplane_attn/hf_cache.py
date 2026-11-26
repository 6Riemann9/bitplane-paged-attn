"""
面向 HuggingFace `transformers` (`Qwen3` / `Qwen3.5` Hybrid) 的真端到端省显存 `BitPlaneDynamicCache`。

核心机制：
  1. 原地封页并即时释放全精历史：Prefill 与 Decode 阶段通过 `PagedBitPlaneKVPool`
     将已满页的历史 KV 增量压缩为 4-to-1 uint8 物理页池并克隆保留极小全精尾窗，
     不再于 `CacheLayer.keys / values` 中长期驻留全精历史 KV 张量；
  2. 原生混合架构兼容：自动识别 `Qwen3.5` 的 `full_attention` 与 `linear_attention`
     （`GatedDeltaNet`），仅替换全局注意力层为 `BitPlaneCacheLayer`，线性层沿用原生 `LinearAttentionLayer`；
  3. 上下文管理器 `cache.attach(model)`：在 Decode (`L_q == 1`) 无缝分派到真物理页表
     `triton_paged_bitplane_dualstream_decode` / `triton_paged_int4_decode`，在 Prefill (`L_q > 1`)
     自动调用零 $O(L^2)$ 矩阵驻留的 `sdpa_attention_forward`。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import threading
from typing import Any, Iterator
from weakref import WeakKeyDictionary

import torch
from transformers.cache_utils import (
    DYNAMIC_LAYER_TYPE_MAPPING,
    Cache,
    CacheLayerMixin,
    get_layer_types_and_kwargs,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .packed_cache import PagedBitPlaneKVPool, PagedKVBatch
from .quant_bitplane import QuantizedBitPlanePages, paged_bitplane_attention_decode
from .triton_kernels import (
    triton_paged_bitplane_dualstream_decode,
    triton_paged_int4_decode,
)

_ACTIVE_BITPLANE_CACHE: ContextVar[BitPlaneDynamicCache | None] = ContextVar(
    "_ACTIVE_BITPLANE_CACHE",
    default=None,
)
_ATTACH_LOCK = threading.RLock()
_MODEL_PATCH_STATE: WeakKeyDictionary[torch.nn.Module, dict[str, Any]] = WeakKeyDictionary()


class BitPlaneCacheLayer(CacheLayerMixin):
    """
    继承自 `transformers.cache_utils.CacheLayerMixin` 的真分页双位面缓存层。

    不长期保留完整历史全精 KV 张量（`self.keys = None; self.values = None`），
    而是随序列增长实时通过 `PagedBitPlaneKVPool` 增量封存完整物理页并回收旧缓冲区。
    """

    is_sliding = False
    is_croppable = False
    supports_early_init = False

    def __init__(
        self,
        page_size: int = 32,
        recent_tokens: int = 32,
        initial_capacity: int = 8,
    ) -> None:
        super().__init__()
        self.page_size = int(page_size)
        self.recent_tokens = int(recent_tokens)
        self.initial_capacity = int(initial_capacity)
        self.pool: PagedBitPlaneKVPool | None = None
        self.seq_ids: list[int] = []
        self.total_seq_len: int = 0
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.num_kv_heads: int = 0
        self.head_dim: int = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """首次前向时按输入 KV 的真实形状与设备初始化物理页池与并发序列槽。"""
        bsz, h_kv, l_new, head_dim = key_states.shape
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.num_kv_heads = int(h_kv)
        self.head_dim = int(head_dim)

        need_pages = bsz * (max(0, l_new - self.recent_tokens) // self.page_size)
        est_pages = max(self.initial_capacity, need_pages + max(4 * bsz, need_pages // 8))
        self.pool = PagedBitPlaneKVPool(
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            capacity=est_pages,
            kv_dtype=self.dtype,
            device=self.device,
        )
        self.seq_ids = [self.pool.allocate_sequence() for _ in range(bsz)]
        self.keys = None
        self.values = None
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        增量追加 KV 状态至物理页池并在满页时实时压缩释放全精历史：
          - 首轮 Prefill (`prev_len == 0`)：原地封页后直接传回当前 `(key_states, value_states)` 供 Prefill 算完释放；
          - 单步 Decode (`l_new == 1`)：零历史解包、零额外显存拷贝；
          - 续轮多 token (`prev_len > 0` 且 `l_new > 1`)：临时解包历史 4-bit + 尾部窗口供 Prefill 算子消费后即弃。
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        assert self.pool is not None
        prev_len = self.total_seq_len
        l_new = int(key_states.shape[2])

        self.pool.append_batch_kv(
            self.seq_ids,
            key_states,
            value_states,
            recent_tokens=self.recent_tokens,
        )
        self.total_seq_len += l_new

        if l_new == 1 or prev_len == 0:
            return key_states, value_states

        # 多轮 chunk 续写且挂接标准 SDPA Prefill 时：临时组装全序列视图（算完即被垃圾回收，不保留引用）
        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []
        for sid in self.seq_ids:
            u = self.pool.unpack_sequence(sid)
            k_full_s = torch.cat([u["k_int4"].to(self.dtype), u["k_tail"].to(self.dtype)], dim=2)
            v_full_s = torch.cat([u["v_int4"].to(self.dtype), u["v_tail"].to(self.dtype)], dim=2)
            k_chunks.append(k_full_s)
            v_chunks.append(v_full_s)
        return torch.cat(k_chunks, dim=0), torch.cat(v_chunks, dim=0)

    def build_batch(self) -> PagedKVBatch:
        """构建 `PagedKVBatch` 供真页表 Triton 核消费（不在层内长期持有拷贝）。"""
        if self.pool is None:
            raise RuntimeError("BitPlaneCacheLayer 尚未初始化，无法构建 PagedKVBatch")
        return self.pool.build_batch(self.seq_ids)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """返回掩码生成器所需 `(kv_length, kv_offset)`。"""
        return self.total_seq_len + int(query_length), 0

    def get_seq_length(self) -> int:
        """返回逻辑完整序列长度。"""
        return self.total_seq_len

    def get_max_length(self) -> int:
        """动态物理池无静态长度上限，按规范返回 `-1`。"""
        return -1

    def reset(self) -> None:
        """清空物理页池占用并将序列状态重置为未初始化。"""
        if self.pool is not None:
            for sid in list(self.seq_ids):
                self.pool.free_sequence(sid)
        self.pool = None
        self.seq_ids = []
        self.total_seq_len = 0
        self.keys = None
        self.values = None
        self.is_initialized = False

    def offload(self) -> None:
        """物理池默认常驻 GPU；此保留接口防 `Cache` 基类误操作 `self.keys`。"""
        return None

    def prefetch(self) -> None:
        """保留接口以兼容 `Cache` 基类生命周期调用。"""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """支持束搜索按 `beam_idx` 对并发物理序列指针重排与 Prefix Caching 引用迁移。"""
        if not self.is_initialized or self.pool is None or not self.seq_ids:
            return
        old_ids = list(self.seq_ids)
        new_ids = [self.pool.fork_sequence(old_ids[int(idx.item())]) for idx in beam_idx]
        for old_sid in old_ids:
            self.pool.free_sequence(old_sid)
        self.seq_ids = new_ids

    def batch_repeat_interleave(self, repeats: int) -> None:
        """将当前并发批内每条序列按 `repeats` 零拷贝分叉扩增物理块引用。"""
        if not self.is_initialized or self.pool is None or repeats <= 1:
            return
        old_ids = list(self.seq_ids)
        new_ids: list[int] = []
        for sid in old_ids:
            for r in range(repeats):
                new_ids.append(sid if r == 0 else self.pool.fork_sequence(sid))
        self.seq_ids = new_ids

    def memory_stats(self) -> dict[str, Any]:
        """返回本层物理页池真实显存统计字典。"""
        if not self.is_initialized or self.pool is None:
            return {
                "unique_active_blocks": 0,
                "logical_total_blocks": 0,
                "physical_pool_active_bytes": 0,
                "physical_pool_reserved_bytes": 0,
                "tail_bytes": 0,
                "total_active_bytes": 0,
                "logical_fp16_equiv_bytes": 0,
                "effective_compression_ratio": 1.0,
            }
        return self.pool.memory_stats()


def _has_masked_positions(attention_mask: torch.Tensor | None) -> bool:
    """检测 Decode 传入的掩码中是否实际存在被屏蔽位置；无掩码时零同步直接走高速 Triton 热路径。"""
    if attention_mask is None:
        return False
    if attention_mask.dtype == torch.bool:
        return bool((~attention_mask).any().item())
    return bool((attention_mask < -1e-3).any().item())


def _pool_seq_to_quantized_pages(
    pool: PagedBitPlaneKVPool,
    seq_id: int,
) -> tuple[QuantizedBitPlanePages, torch.Tensor, torch.Tensor, int]:
    """将物理页池中的单条序列解包为供带掩码 Decode 回退路径使用的量化页与全序列 K/V。"""
    u = pool.unpack_sequence(seq_id)
    seq = pool.sequences[seq_id]
    n_pages = seq.num_pages
    hist_len = n_pages * pool.page_size
    if n_pages == 0:
        empty_m = torch.empty((1, pool.num_kv_heads, 0, pool.head_dim), dtype=torch.float32, device=pool.device)
        pages = QuantizedBitPlanePages(
            page_size=pool.page_size,
            hist_len=0,
            num_pages=0,
            k_base=u["k_base"],
            k_int4=u["k_int4"],
            k_int2=u["k_base"],
            v_base=u["v_base"],
            v_int4=u["v_int4"],
            v_int2=u["v_base"],
            k_page_min=empty_m,
            k_page_max=empty_m,
        )
    else:
        idx_t = torch.tensor(seq.block_ids, dtype=torch.long, device=pool.device)
        km = pool.k_min_pool.index_select(0, idx_t).float().permute(1, 0, 2).unsqueeze(0)
        ks = pool.k_scale_pool.index_select(0, idx_t).float().permute(1, 0, 2).unsqueeze(0)
        pages = QuantizedBitPlanePages(
            page_size=pool.page_size,
            hist_len=hist_len,
            num_pages=n_pages,
            k_base=u["k_base"],
            k_int4=u["k_int4"],
            k_int2=u["k_base"],
            v_base=u["v_base"],
            v_int4=u["v_int4"],
            v_int2=u["v_base"],
            k_page_min=km,
            k_page_max=km + 15.0 * ks,
        )
    k_full = torch.cat([u["k_int4"], u["k_tail"]], dim=2)
    v_full = torch.cat([u["v_int4"], u["v_tail"]], dim=2)
    return pages, k_full, v_full, hist_len


def _masked_bitplane_decode_fallback(
    query: torch.Tensor,
    layer: BitPlaneCacheLayer,
    attention_mask: torch.Tensor,
    scaling: float,
    active: BitPlaneDynamicCache,
) -> torch.Tensor:
    """
    当 Decode (`L_q == 1`) 收到 Left-padding 或自定义 Mask 时，按位面精度等价反量化执行精确带掩码解码，
    保证 HuggingFace `attention_mask` 语义不静默失效。
    """
    if layer.pool is None:
        raise RuntimeError("BitPlaneCacheLayer 尚未初始化，无法执行带掩码回退解码")
    bsz = int(query.shape[0])
    neg_inf = torch.finfo(torch.float32).min
    ref_mode = "int4" if active.mode == "triton_int4" else "bitplane_dualstream"
    outs: list[torch.Tensor] = []
    bits_list: list[float] = []

    for b in range(bsz):
        sid = layer.seq_ids[b]
        seq = layer.pool.sequences[sid]
        pages_b, k_full_b, v_full_b, hist_len_b = _pool_seq_to_quantized_pages(layer.pool, sid)
        total_len_b = int(k_full_b.shape[2])

        if attention_mask.ndim == 4:
            mask_b = attention_mask[0:1] if attention_mask.shape[0] == 1 else attention_mask[b : b + 1]
        elif attention_mask.ndim == 2:
            m2 = attention_mask[0:1] if attention_mask.shape[0] == 1 else attention_mask[b : b + 1]
            mask_b = m2[:, None, None, :]
        else:
            raise ValueError(f"不支持的 attention_mask 维度: {tuple(attention_mask.shape)}")

        if mask_b.shape[-1] > total_len_b:
            mask_b = mask_b[..., :total_len_b]
        if mask_b.dtype == torch.bool:
            mask_b = torch.where(mask_b, 0.0, neg_inf).to(torch.float32)
        else:
            mask_b = mask_b.to(torch.float32)

        out_b, st_b = paged_bitplane_attention_decode(
            query=query[b : b + 1],
            key=k_full_b,
            value=v_full_b,
            scaling=scaling,
            attention_mask=mask_b,
            mode=ref_mode,
            page_size=layer.pool.page_size,
            recent_tokens=max(1, seq.tail_len),
            hist_len=hist_len_b,
            pages_cache=pages_b,
            hot_ratio=active.hot_ratio,
            warm_ratio=active.warm_ratio,
            ellipsoid_beta=active.beta,
            pin_page0=active.pin_page0,
            score_mode=active.score_mode,
        )
        outs.append(out_b)
        if active.collect_stats:
            bits_list.append(float(st_b["logical_read_bits_per_kv_elem"]))

    if active.collect_stats and bits_list:
        active.read_bits_samples.append(sum(bits_list) / len(bits_list))
    return torch.cat(outs, dim=0).to(query.dtype)


def _bitplane_paged_attention_interface(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    注册于 HuggingFace `ALL_ATTENTION_FUNCTIONS["bitplane_paged"]` 的端到端注意力分派入口：
      - Prefill (`L_q > 1`)：委托原生 `sdpa_attention_forward`，避免生成 $O(L^2)$ Eager 显存尖峰；
      - Decode (`L_q == 1`)：无屏蔽位时直通零分支物理页表 Triton 核；存在 Left-padding/自定义掩码时
        按 `strict_mask` 显式报错或走高精度等价回退，确保 `attention_mask` 语义完整。
    """
    active = _ACTIVE_BITPLANE_CACHE.get()
    if active is None or query.shape[2] != 1:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    l_idx = int(getattr(module, "layer_idx", -1))
    if l_idx < 0 or l_idx >= len(active.layers):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    layer = active.layers[l_idx]
    if not isinstance(layer, BitPlaneCacheLayer):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    if attention_mask is not None and _has_masked_positions(attention_mask):
        if active.strict_mask:
            raise NotImplementedError(
                "BitPlaneDynamicCache 当前处于 strict_mask=True 模式，不支持含非平凡屏蔽位的 Decode Mask。"
            )
        out = _masked_bitplane_decode_fallback(query, layer, attention_mask, scaling, active)
        return out, None

    batch = layer.build_batch()
    if active.mode == "triton_dualstream":
        out, st = triton_paged_bitplane_dualstream_decode(
            query=query,
            batch=batch,
            scaling=scaling,
            hot_ratio=active.hot_ratio,
            warm_ratio=active.warm_ratio,
            beta=active.beta,
            score_mode=active.score_mode,
            pin_page0=active.pin_page0,
            return_stats=True,
        )
        if active.collect_stats:
            active.read_bits_samples.append(float(st["logical_read_bits_per_kv_elem"]))
    elif active.mode == "triton_int4":
        out = triton_paged_int4_decode(query=query, batch=batch, scaling=scaling)
        if active.collect_stats:
            from .quant_bitplane import compute_logical_read_bits_per_kv_elem

            active.read_bits_samples.append(
                compute_logical_read_bits_per_kv_elem(
                    "int4", head_dim=int(query.shape[-1]), page_size=active.page_size
                )
            )
    else:
        raise ValueError(f"未知的 BitPlaneDynamicCache 解码模式: {active.mode}")

    # Triton 核直接输出 HuggingFace 期望布局 (B, L_q=1, H_q, d)
    return out, None


ALL_ATTENTION_FUNCTIONS.register("bitplane_paged", _bitplane_paged_attention_interface)
if "sdpa" in ALL_MASK_ATTENTION_FUNCTIONS:
    ALL_MASK_ATTENTION_FUNCTIONS.register("bitplane_paged", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])


class BitPlaneDynamicCache(Cache):
    """
    面向 `Qwen3` / `Qwen3.5`（含 Hybrid 3:1 `GatedDeltaNet`）等大模型的即插即用省显存分页双位面 Cache。

    用法示例：
        ```python
        cache = BitPlaneDynamicCache(config=model.config, hot_ratio=0.15, warm_ratio=0.35)
        with cache.attach(model):
            outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=128)
        print(cache.memory_stats())
        ```
    """

    def __init__(
        self,
        config: Any | None = None,
        page_size: int = 32,
        recent_tokens: int = 32,
        initial_capacity: int = 8,
        mode: str = "triton_dualstream",
        hot_ratio: float = 0.15,
        warm_ratio: float = 0.35,
        beta: float = 1.5,
        score_mode: str = "anisotropic",
        pin_page0: bool = True,
        collect_stats: bool = False,
        strict_mask: bool = False,
    ) -> None:
        self.page_size = int(page_size)
        self.recent_tokens = int(recent_tokens)
        self.initial_capacity = int(initial_capacity)
        self.mode = str(mode)
        self.hot_ratio = float(hot_ratio)
        self.warm_ratio = float(warm_ratio)
        self.beta = float(beta)
        self.score_mode = str(score_mode)
        self.pin_page0 = bool(pin_page0)
        self.collect_stats = bool(collect_stats)
        self.strict_mask = bool(strict_mask)
        self.read_bits_samples: list[float] = []

        layers: list[Any] = []
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            layer_types, per_layer_kwargs = get_layer_types_and_kwargs(decoder_config)
            for layer_type, layer_kwargs in zip(layer_types, per_layer_kwargs):
                if layer_type == "full_attention":
                    layers.append(
                        BitPlaneCacheLayer(
                            page_size=self.page_size,
                            recent_tokens=self.recent_tokens,
                            initial_capacity=self.initial_capacity,
                        )
                    )
                else:
                    layers.append(DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs))

        if layers:
            super().__init__(layers=layers)
        else:
            super().__init__(
                layer_class_to_replicate=lambda: BitPlaneCacheLayer(
                    page_size=self.page_size,
                    recent_tokens=self.recent_tokens,
                    initial_capacity=self.initial_capacity,
                )
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """自动追踪当前正在前向写入的 Cache 实例（防止 `copy.deepcopy(cache)` 后活跃指针仍指向旧对象）。"""
        active = _ACTIVE_BITPLANE_CACHE.get()
        if active is not None and active is not self:
            _ACTIVE_BITPLANE_CACHE.set(self)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    @contextmanager
    def attach(self, model: torch.nn.Module) -> Iterator[BitPlaneDynamicCache]:
        """
        使用进程内带锁引用计数安全切换共享 `model` 的 `_attn_implementation` 为 `"bitplane_paged"`，
        并激活当前上下文的 `BitPlaneDynamicCache`。仅当最外层/最后一个并发持有者退出时才还原原配置，
        防止交错并发请求恢复顺序冲突。
        """
        with _ATTACH_LOCK:
            state = _MODEL_PATCH_STATE.get(model)
            if state is None:
                saved_cfg: list[tuple[Any, Any]] = []
                seen_ids: set[int] = set()

                def _collect_and_set(obj: Any) -> None:
                    if not hasattr(obj, "config") or obj.config is None:
                        return
                    for cfg in (obj.config, getattr(obj.config, "text_config", None)):
                        if cfg is not None and hasattr(cfg, "_attn_implementation") and id(cfg) not in seen_ids:
                            seen_ids.add(id(cfg))
                            saved_cfg.append((cfg, cfg._attn_implementation))
                            cfg._attn_implementation = "bitplane_paged"

                _collect_and_set(model)
                for mod in model.modules():
                    _collect_and_set(mod)
                _MODEL_PATCH_STATE[model] = {"ref_count": 1, "saved_cfg": saved_cfg}
            else:
                state["ref_count"] += 1

        token = _ACTIVE_BITPLANE_CACHE.set(self)
        try:
            yield self
        finally:
            _ACTIVE_BITPLANE_CACHE.reset(token)
            with _ATTACH_LOCK:
                state = _MODEL_PATCH_STATE.get(model)
                if state is not None:
                    state["ref_count"] -= 1
                    if state["ref_count"] <= 0:
                        _MODEL_PATCH_STATE.pop(model, None)
                        for cfg_obj, old_impl in reversed(state["saved_cfg"]):
                            cfg_obj._attn_implementation = old_impl

    @contextmanager
    def activate(self) -> Iterator[BitPlaneDynamicCache]:
        """
        无状态激活当前 `BitPlaneDynamicCache` 上下文，不触碰共享 `model.config`。
        推荐先对模型执行一次 `model.set_attn_implementation("bitplane_paged")`，随后高并发请求均使用 `with cache.activate():`。
        """
        token = _ACTIVE_BITPLANE_CACHE.set(self)
        try:
            yield self
        finally:
            _ACTIVE_BITPLANE_CACHE.reset(token)

    def memory_stats(self) -> dict[str, Any]:
        """聚合统计全部全局注意力层物理页池的真实 GPU 显存占用与相对 BF16/FP16 压缩比。"""
        bp_layers = [lyr for lyr in self.layers if isinstance(lyr, BitPlaneCacheLayer)]
        total_active_bytes = 0
        total_reserved_bytes = 0
        total_tail_bytes = 0
        total_equiv_bytes = 0
        unique_blocks = 0
        logical_blocks = 0
        for lyr in bp_layers:
            st = lyr.memory_stats()
            total_active_bytes += int(st["total_active_bytes"])
            total_reserved_bytes += int(st["physical_pool_reserved_bytes"])
            total_tail_bytes += int(st["tail_bytes"])
            total_equiv_bytes += int(st["logical_fp16_equiv_bytes"])
            unique_blocks += int(st["unique_active_blocks"])
            logical_blocks += int(st["logical_total_blocks"])

        return {
            "num_paged_layers": len(bp_layers),
            "unique_active_blocks": unique_blocks,
            "logical_total_blocks": logical_blocks,
            "total_active_bytes": total_active_bytes,
            "total_reserved_bytes": total_reserved_bytes,
            "tail_bytes": total_tail_bytes,
            "logical_fp16_equiv_bytes": total_equiv_bytes,
            "active_mb": total_active_bytes / (1024.0 ** 2),
            "reserved_mb": (total_reserved_bytes + total_tail_bytes) / (1024.0 ** 2),
            "fp16_equiv_mb": total_equiv_bytes / (1024.0 ** 2),
            "compression_ratio": total_equiv_bytes / max(total_active_bytes, 1),
            "reserved_compression_ratio": total_equiv_bytes / max(total_reserved_bytes + total_tail_bytes, 1),
        }
