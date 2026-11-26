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
from typing import Any, Iterator

import torch
from transformers.cache_utils import (
    DYNAMIC_LAYER_TYPE_MAPPING,
    Cache,
    CacheLayerMixin,
    get_layer_types_and_kwargs,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .packed_cache import PagedBitPlaneKVPool, PagedKVBatch
from .triton_kernels import (
    triton_paged_bitplane_dualstream_decode,
    triton_paged_int4_decode,
)

_ACTIVE_BITPLANE_CACHE: ContextVar[BitPlaneDynamicCache | None] = ContextVar(
    "_ACTIVE_BITPLANE_CACHE",
    default=None,
)


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
      - Decode (`L_q == 1`)：从当前上下文活跃的 `BitPlaneDynamicCache` 读取真物理页池 `PagedKVBatch`
        并直接运行无分支双流 / INT4 Triton GPU 解码核。
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
        将 `model` 的注意力实现临时切换为 `"bitplane_paged"` 并激活当前分页双位面上下文，
        退出 `with` 代码块后精确恢复原 `_attn_implementation`。
        """
        saved_cfg: list[tuple[Any, Any]] = []

        def _set_impl(obj: Any, impl: str) -> None:
            if hasattr(obj, "config") and obj.config is not None:
                if hasattr(obj.config, "_attn_implementation"):
                    saved_cfg.append((obj.config, obj.config._attn_implementation))
                    obj.config._attn_implementation = impl
                tc = getattr(obj.config, "text_config", None)
                if tc is not None and hasattr(tc, "_attn_implementation"):
                    saved_cfg.append((tc, tc._attn_implementation))
                    tc._attn_implementation = impl

        _set_impl(model, "bitplane_paged")
        for mod in model.modules():
            _set_impl(mod, "bitplane_paged")

        token = _ACTIVE_BITPLANE_CACHE.set(self)
        try:
            yield self
        finally:
            _ACTIVE_BITPLANE_CACHE.reset(token)
            for cfg_obj, old_impl in reversed(saved_cfg):
                cfg_obj._attn_implementation = old_impl

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
