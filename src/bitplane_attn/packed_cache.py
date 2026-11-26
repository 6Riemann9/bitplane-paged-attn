"""
面向 PagedAttention 与 Prefix Caching 的 4-to-1 紧缩物理位面 KV-Cache 模块。

核心设计：
  1. 零洗牌 4 象限紧缩打包（Zero-Shuffle Quadrant 4-to-1 Packing）：
     将头维 d（4 的整数倍，如 64 / 128 / 256）等分为 4 个连续子块 [0:D4], [D4:2D4], [2D4:3D4], [3D4:d]，
     在同一位置 j 上将 4 个 2-bit 整数压入 1 个 uint8 字节（bits [1:0], [3:2], [5:4], [7:6]）。
     在 Triton GPU 核内对 4 个象限均做 100% 连续合并访存，零跨步、零字节内重排洗牌。
  2. 双级封装形态：
     - `PackedBitPlaneKVCache`：页优先连续张量（兼容既有单核基准与参考校验）；
     - `PagedBitPlaneKVPool` / `PagedKVBatch`：生产级 `PagedAttention` 物理页池，包含
       Block-First 4D 物理存储池、逻辑页到物理页映射表 (`block_table`)、动态增量封页、
       Prefix Caching 引用计数与 Copy-on-Write (CoW)、以及不等长 Continuous Batching。
  3. 尾部全精窗口支持独立微张量存储，避免保留大型 Prefill 缓冲区引用导致的隐性显存滞留。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

import torch


def pack_2bit_quadrants(q2: torch.Tensor) -> torch.Tensor:
    """
    将取值范围为 {0, 1, 2, 3} 的最后一维按 4 象限零洗牌紧缩打包进 uint8。

    Args:
        q2: 形状 (..., d) 的 uint8 张量，要求 d % 4 == 0 且值在 [0, 3]。

    Returns:
        形状 (..., d // 4) 的 uint8 连续紧缩张量。
    """
    head_dim = q2.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError(f"头维 d={head_dim} 必须为 4 的整数倍才能执行 4 象限 2-bit 紧缩打包")
    d4 = head_dim // 4
    parts = q2.to(torch.uint8).reshape(*q2.shape[:-1], 4, d4)
    packed = (
        parts[..., 0, :]
        | (parts[..., 1, :] << 2)
        | (parts[..., 2, :] << 4)
        | (parts[..., 3, :] << 6)
    )
    return packed.contiguous()


def unpack_2bit_quadrants(packed: torch.Tensor) -> torch.Tensor:
    """
    将 4 象限紧缩打包的 uint8 张量无损解包为完整头维 2-bit 整数张量。

    Args:
        packed: 形状 (..., D4) 的 uint8 张量。

    Returns:
        形状 (..., D4 * 4) 的 uint8 张量，值域为 {0, 1, 2, 3}。
    """
    q0 = packed & 3
    q1 = (packed >> 2) & 3
    q2 = (packed >> 4) & 3
    q3 = (packed >> 6) & 3
    return torch.cat([q0, q1, q2, q3], dim=-1).contiguous()


@dataclass
class PackedBitPlaneKVCache:
    """
    存储物理紧缩的双位面历史整页 KV 缓存及全精尾部窗口。

    历史整页区间 [0, hist_len) 存储为硬件紧凑的 uint8 4-to-1 双位面 + float16 页尺度；
    尾部区间 [hist_len, L_total) 保留原始高精 dtype（BF16/FP16）以护住最近上下文。
    """

    page_size: int
    hist_len: int
    num_pages: int
    batch_size: int
    num_kv_heads: int
    head_dim: int
    kv_dtype: torch.dtype
    # 历史整页物理紧缩张量 (P == num_pages)
    k_base_packed: torch.Tensor  # (B, H_kv, P, S, D4) uint8
    k_res_packed: torch.Tensor   # (B, H_kv, P, S, D4) uint8
    k_min: torch.Tensor          # (B, H_kv, P, d) float16
    k_scale: torch.Tensor        # (B, H_kv, P, d) float16
    v_base_packed: torch.Tensor  # (B, H_kv, P, S, D4) uint8
    v_res_packed: torch.Tensor   # (B, H_kv, P, S, D4) uint8
    v_min: torch.Tensor          # (B, H_kv, P, S) float16
    v_scale: torch.Tensor        # (B, H_kv, P, S) float16
    # 尾部未压缩全精窗口 (L_tail = L_total - hist_len)
    k_tail: torch.Tensor         # (B, H_kv, L_tail, d)
    v_tail: torch.Tensor         # (B, H_kv, L_tail, d)

    @property
    def total_seq_len(self) -> int:
        """返回逻辑完整序列长度 (hist_len + L_tail)。"""
        return self.hist_len + int(self.k_tail.shape[2])

    @property
    def tail_len(self) -> int:
        """返回未压缩尾部全精序列长度。"""
        return int(self.k_tail.shape[2])

    @classmethod
    def from_kv(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        page_size: int = 32,
        recent_tokens: int = 32,
        hist_len: int | None = None,
    ) -> PackedBitPlaneKVCache:
        """
        从全精 Key/Value 张量在 GPU 上一次性向量化切分并紧缩打包为页级双位面缓存。

        Args:
            key: 形状 (B, H_kv, L, d)
            value: 形状 (B, H_kv, L, d)
            page_size: 物理页大小，默认 32
            recent_tokens: 当 hist_len 为 None 时，末尾保留全精的最小 token 数
            hist_len: 可显式指定历史压缩前缀长度（将向下对齐到 page_size 整数倍）
        """
        if key.shape != value.shape:
            raise ValueError(f"Key 形状 {key.shape} 与 Value 形状 {value.shape} 不一致")
        bsz, h_kv, seq_len, head_dim = key.shape
        if head_dim % 4 != 0:
            raise ValueError(f"头维 d={head_dim} 必须为 4 的整数倍")
        d4 = head_dim // 4

        if hist_len is None:
            safe_upper = max(0, seq_len - recent_tokens)
            hist_len_eff = (safe_upper // page_size) * page_size
        else:
            hist_len_eff = (max(0, min(int(hist_len), seq_len)) // page_size) * page_size

        num_pages = hist_len_eff // page_size
        device = key.device

        if num_pages == 0:
            empty_u8 = torch.empty((bsz, h_kv, 0, page_size, d4), dtype=torch.uint8, device=device)
            empty_kd = torch.empty((bsz, h_kv, 0, head_dim), dtype=torch.float16, device=device)
            empty_vs = torch.empty((bsz, h_kv, 0, page_size), dtype=torch.float16, device=device)
            return cls(
                page_size=page_size,
                hist_len=0,
                num_pages=0,
                batch_size=bsz,
                num_kv_heads=h_kv,
                head_dim=head_dim,
                kv_dtype=key.dtype,
                k_base_packed=empty_u8,
                k_res_packed=empty_u8,
                k_min=empty_kd,
                k_scale=empty_kd,
                v_base_packed=empty_u8,
                v_res_packed=empty_u8,
                v_min=empty_vs,
                v_scale=empty_vs,
                k_tail=key.contiguous(),
                v_tail=value.contiguous(),
            )

        # 1) Key 页内 per-channel 嵌套 4-bit 切分 (B, H_kv, P, S, d) 沿 S (dim=-2) 求最值
        k_paged = key[:, :, :hist_len_eff, :].view(bsz, h_kv, num_pages, page_size, head_dim).float()
        k_min_fp32 = k_paged.min(dim=-2).values  # (B, H_kv, P, d)
        k_max_fp32 = k_paged.max(dim=-2).values
        k_scale_fp32 = torch.clamp(k_max_fp32 - k_min_fp32, min=1e-7) / 15.0
        # 使用转成 float16 后的 min/scale 反算整数，保持与推理核读取 fp16 元数据完全同源
        k_min_fp16 = k_min_fp32.to(torch.float16)
        k_scale_fp16 = k_scale_fp32.to(torch.float16)
        k_min_q = k_min_fp16.float().unsqueeze(-2)
        k_scale_q = torch.clamp(k_scale_fp16.float(), min=1e-7).unsqueeze(-2)

        k_q4 = torch.clamp(torch.round((k_paged - k_min_q) / k_scale_q), 0, 15).to(torch.uint8)
        k_base_packed = pack_2bit_quadrants(k_q4 >> 2)
        k_res_packed = pack_2bit_quadrants(k_q4 & 3)

        # 2) Value per-token 嵌套 4-bit 切分 (B, H_kv, P, S, d) 沿 d (dim=-1) 求最值
        v_paged = value[:, :, :hist_len_eff, :].view(bsz, h_kv, num_pages, page_size, head_dim).float()
        v_min_fp32 = v_paged.min(dim=-1).values  # (B, H_kv, P, S)
        v_max_fp32 = v_paged.max(dim=-1).values
        v_scale_fp32 = torch.clamp(v_max_fp32 - v_min_fp32, min=1e-7) / 15.0
        v_min_fp16 = v_min_fp32.to(torch.float16)
        v_scale_fp16 = v_scale_fp32.to(torch.float16)
        v_min_q = v_min_fp16.float().unsqueeze(-1)
        v_scale_q = torch.clamp(v_scale_fp16.float(), min=1e-7).unsqueeze(-1)

        v_q4 = torch.clamp(torch.round((v_paged - v_min_q) / v_scale_q), 0, 15).to(torch.uint8)
        v_base_packed = pack_2bit_quadrants(v_q4 >> 2)
        v_res_packed = pack_2bit_quadrants(v_q4 & 3)

        k_tail = key[:, :, hist_len_eff:, :].contiguous()
        v_tail = value[:, :, hist_len_eff:, :].contiguous()

        return cls(
            page_size=page_size,
            hist_len=hist_len_eff,
            num_pages=num_pages,
            batch_size=bsz,
            num_kv_heads=h_kv,
            head_dim=head_dim,
            kv_dtype=key.dtype,
            k_base_packed=k_base_packed,
            k_res_packed=k_res_packed,
            k_min=k_min_fp16.contiguous(),
            k_scale=k_scale_fp16.contiguous(),
            v_base_packed=v_base_packed,
            v_res_packed=v_res_packed,
            v_min=v_min_fp16.contiguous(),
            v_scale=v_scale_fp16.contiguous(),
            k_tail=k_tail,
            v_tail=v_tail,
        )

    def unpack_to_dense_dequantized(self) -> dict[str, torch.Tensor]:
        """将紧缩打包张量全部解包为浮点矩阵，供高精度测试与参考校验对照。"""
        if self.num_pages == 0:
            empty = torch.empty(
                (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                dtype=torch.float32,
                device=self.k_tail.device,
            )
            return {"k_base": empty, "k_int4": empty, "v_base": empty, "v_int4": empty}

        bsz, h_kv, p, s, _ = self.k_base_packed.shape
        k_hi = unpack_2bit_quadrants(self.k_base_packed).float()
        k_lo = unpack_2bit_quadrants(self.k_res_packed).float()
        k_m = self.k_min.float().unsqueeze(-2)
        k_sc = self.k_scale.float().unsqueeze(-2)
        k_base_5d = k_m + (4.0 * k_hi + 1.5) * k_sc
        k_int4_5d = k_base_5d + (k_lo - 1.5) * k_sc
        k_base = k_base_5d.view(bsz, h_kv, p * s, self.head_dim)
        k_int4 = k_int4_5d.view(bsz, h_kv, p * s, self.head_dim)

        v_hi = unpack_2bit_quadrants(self.v_base_packed).float()
        v_lo = unpack_2bit_quadrants(self.v_res_packed).float()
        v_m = self.v_min.float().unsqueeze(-1)
        v_sc = self.v_scale.float().unsqueeze(-1)
        v_base_5d = v_m + (4.0 * v_hi + 1.5) * v_sc
        v_int4_5d = v_base_5d + (v_lo - 1.5) * v_sc
        v_base = v_base_5d.view(bsz, h_kv, p * s, self.head_dim)
        v_int4 = v_int4_5d.view(bsz, h_kv, p * s, self.head_dim)

        return {"k_base": k_base, "k_int4": k_int4, "v_base": v_base, "v_int4": v_int4}

    def memory_stats(self) -> dict[str, Any]:
        """精确核算全部物理显存字节开销与相对于 FP16/BF16 的实际压缩倍率。"""
        def _nbytes(t: torch.Tensor) -> int:
            return int(t.numel() * t.element_size())

        k_base_b = _nbytes(self.k_base_packed)
        k_res_b = _nbytes(self.k_res_packed)
        v_base_b = _nbytes(self.v_base_packed)
        v_res_b = _nbytes(self.v_res_packed)
        base_payload_b = k_base_b + v_base_b
        res_payload_b = k_res_b + v_res_b
        metadata_b = _nbytes(self.k_min) + _nbytes(self.k_scale) + _nbytes(self.v_min) + _nbytes(self.v_scale)
        tail_b = _nbytes(self.k_tail) + _nbytes(self.v_tail)
        total_b = base_payload_b + res_payload_b + metadata_b + tail_b

        total_elems = 2 * self.batch_size * self.num_kv_heads * self.total_seq_len * self.head_dim
        fp16_equiv_b = total_elems * 2
        pages_only_b = base_payload_b + res_payload_b + metadata_b
        pages_elems = 2 * self.batch_size * self.num_kv_heads * self.hist_len * self.head_dim

        return {
            "base_payload_bytes": base_payload_b,
            "res_payload_bytes": res_payload_b,
            "metadata_bytes": metadata_b,
            "tail_bytes": tail_b,
            "total_bytes": total_b,
            "fp16_equiv_bytes": fp16_equiv_b,
            "compression_ratio": fp16_equiv_b / max(total_b, 1),
            "effective_bits_stored_total": (total_b * 8.0) / max(total_elems, 1),
            "effective_bits_stored_pages": (pages_only_b * 8.0) / max(pages_elems, 1),
            "base_only_vram_ratio": fp16_equiv_b / max(base_payload_b + metadata_b + tail_b, 1),
        }


def _quantize_pages_to_packed(
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    对以物理页为第一维的整页 KV 张量执行硬件紧缩嵌套 4-bit 双位面切分与打包。

    Args:
        key_pages: 形状 `(N_pages, H_kv, S, d)`。
        value_pages: 形状 `(N_pages, H_kv, S, d)`。

    Returns:
        包含 8 个物理池切片张量的字典（`k_base`, `k_res`, `k_min`, `k_scale`,
        `v_base`, `v_res`, `v_min`, `v_scale`）。
    """
    if key_pages.shape != value_pages.shape:
        raise ValueError(f"整页 Key 形状 {key_pages.shape} 与 Value 形状 {value_pages.shape} 不一致")
    if key_pages.ndim != 4:
        raise ValueError(f"期望 4D 张量 (N_pages, H_kv, S, d)，实际收到 {key_pages.ndim}D")
    head_dim = key_pages.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError(f"头维 d={head_dim} 必须为 4 的整数倍才能执行 4 象限打包")

    # Key：页内 per-channel（沿 S 维 dim=-2）量化
    k_fp32 = key_pages.float()
    k_min_fp32 = k_fp32.min(dim=-2).values
    k_max_fp32 = k_fp32.max(dim=-2).values
    k_scale_fp32 = torch.clamp(k_max_fp32 - k_min_fp32, min=1e-7) / 15.0
    k_min_fp16 = k_min_fp32.to(torch.float16)
    k_scale_fp16 = k_scale_fp32.to(torch.float16)
    k_min_q = k_min_fp16.float().unsqueeze(-2)
    k_scale_q = torch.clamp(k_scale_fp16.float(), min=1e-7).unsqueeze(-2)
    k_q4 = torch.clamp(torch.round((k_fp32 - k_min_q) / k_scale_q), 0, 15).to(torch.uint8)
    k_base = pack_2bit_quadrants(k_q4 >> 2)
    k_res = pack_2bit_quadrants(k_q4 & 3)

    # Value：per-token（沿 head_dim 维 dim=-1）量化
    v_fp32 = value_pages.float()
    v_min_fp32 = v_fp32.min(dim=-1).values
    v_max_fp32 = v_fp32.max(dim=-1).values
    v_scale_fp32 = torch.clamp(v_max_fp32 - v_min_fp32, min=1e-7) / 15.0
    v_min_fp16 = v_min_fp32.to(torch.float16)
    v_scale_fp16 = v_scale_fp32.to(torch.float16)
    v_min_q = v_min_fp16.float().unsqueeze(-1)
    v_scale_q = torch.clamp(v_scale_fp16.float(), min=1e-7).unsqueeze(-1)
    v_q4 = torch.clamp(torch.round((v_fp32 - v_min_q) / v_scale_q), 0, 15).to(torch.uint8)
    v_base = pack_2bit_quadrants(v_q4 >> 2)
    v_res = pack_2bit_quadrants(v_q4 & 3)

    return {
        "k_base": k_base,
        "k_res": k_res,
        "k_min": k_min_fp16.contiguous(),
        "k_scale": k_scale_fp16.contiguous(),
        "v_base": v_base,
        "v_res": v_res,
        "v_min": v_min_fp16.contiguous(),
        "v_scale": v_scale_fp16.contiguous(),
    }


@dataclass
class PagedSequenceState:
    """
    单条并发请求在 `PagedBitPlaneKVPool` 中的逻辑状态。

    记录从逻辑页号到物理池块号的映射 `block_ids`，以及未满页的全精尾部缓冲区 `(k_tail, v_tail)`。
    """

    seq_id: int
    page_size: int
    block_ids: list[int] = field(default_factory=list)
    k_tail: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    v_tail: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    @property
    def num_pages(self) -> int:
        """返回该序列已封存的逻辑整页数。"""
        return len(self.block_ids)

    @property
    def tail_len(self) -> int:
        """返回该序列当前全精尾部窗口 token 数。"""
        if self.k_tail.numel() == 0:
            return 0
        return int(self.k_tail.shape[2])

    @property
    def total_seq_len(self) -> int:
        """返回该序列的逻辑总长度（历史整页 tokens + 尾部 tokens）。"""
        return self.num_pages * self.page_size + self.tail_len


@dataclass
class PagedKVBatch:
    """
    面向连续批处理（Continuous Batching）的真 `PagedAttention` GPU 批视图。

    支持同批次内不同序列拥有完全不同的历史页数、非连续/乱序物理块表及不等长尾部窗口；
    同时携带纯 Host 元信息元组以避免热路径触发 GPU->CPU `.item()` 同步。
    """

    pool: PagedBitPlaneKVPool
    block_table: torch.Tensor        # (B, max_num_pages) int32
    num_pages_per_seq: torch.Tensor  # (B,) int32
    k_tail: torch.Tensor             # (B, H_kv, max_tail_len, d)
    v_tail: torch.Tensor             # (B, H_kv, max_tail_len, d)
    tail_lens: torch.Tensor          # (B,) int32
    seq_lens: torch.Tensor           # (B,) int32
    num_pages_host: tuple[int, ...] = ()
    tail_lens_host: tuple[int, ...] = ()
    max_pages_host: int = 0
    max_tail_host: int = 0
    is_uniform_pages: bool = True

    def __post_init__(self) -> None:
        if not self.num_pages_host:
            self.max_pages_host = int(self.block_table.shape[1])
            self.max_tail_host = int(self.k_tail.shape[2])

    @property
    def batch_size(self) -> int:
        """返回当前批次大小。"""
        return int(self.num_pages_per_seq.shape[0])

    @property
    def num_kv_heads(self) -> int:
        """返回 KV 头数。"""
        return self.pool.num_kv_heads

    @property
    def head_dim(self) -> int:
        """返回单头维度。"""
        return self.pool.head_dim

    @property
    def page_size(self) -> int:
        """返回物理页大小。"""
        return self.pool.page_size

    @property
    def max_num_pages(self) -> int:
        """返回批次中最大的已打包整页数。"""
        return self.max_pages_host

    @property
    def max_tail_len(self) -> int:
        """返回批次中最大的尾部全精窗口长度。"""
        return self.max_tail_host

    @property
    def kv_dtype(self) -> torch.dtype:
        """返回尾部全精 KV 的浮点类型（如 BF16 / FP16）。"""
        return self.pool.kv_dtype


class PagedBitPlaneKVPool:
    """
    生产级 Bit-Plane Paged KV-Cache 物理页池与块管理器。

    核心能力：
      1. Block-First 4D 物理池：所有紧缩位面与页元数据均按 `(Capacity, H_kv, ...)` 分配，
         各物理块内部保持 100% 连续合并访存，跨请求按需动态分配与释放；
      2. 自动按需扩容与引用计数：支持 Prefix Caching（零拷贝分叉 `fork_sequence` 与内容哈希命中）
         及 Copy-on-Write (`copy_on_write`)；
      3. 动态增量封页与 Continuous Batching：解码步中全精尾部一攒满 `page_size + recent_tokens`
         即刻自动封存新物理页并克隆缩减尾部，避免隐性持有大型 Prefill Storage。
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 32,
        capacity: int = 64,
        kv_dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda:0",
    ) -> None:
        if head_dim % 4 != 0:
            raise ValueError(f"头维 d={head_dim} 必须为 4 的整数倍")
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.d4 = self.head_dim // 4
        self.page_size = int(page_size)
        self.capacity = max(1, int(capacity))
        self.kv_dtype = kv_dtype
        self.device = torch.device(device)

        self._alloc_pool_tensors(self.capacity)
        # 采用列表尾部 pop 实现 O(1) 分配；初始反转使首次按 0, 1, 2, ... 依次分配
        self.free_blocks: list[int] = list(range(self.capacity - 1, -1, -1))
        self.ref_counts: list[int] = [0] * self.capacity
        self.sequences: dict[int, PagedSequenceState] = {}
        self.prefix_table: dict[bytes, int] = {}
        self._block_prefix_key: dict[int, bytes] = {}
        self._next_seq_id: int = 0
        self._epoch: int = 0
        self._bt_cache_key: tuple[int, tuple[int, ...]] | None = None
        self._bt_cache_val: tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int, bool] | None = None

    def _alloc_pool_tensors(self, cap: int) -> None:
        """按指定物理页容量分配 8 张 Block-First 物理池张量。"""
        h_kv, s, d4, d = self.num_kv_heads, self.page_size, self.d4, self.head_dim
        dev = self.device
        self.k_base_pool = torch.empty((cap, h_kv, s, d4), dtype=torch.uint8, device=dev)
        self.k_res_pool = torch.empty((cap, h_kv, s, d4), dtype=torch.uint8, device=dev)
        self.k_min_pool = torch.empty((cap, h_kv, d), dtype=torch.float16, device=dev)
        self.k_scale_pool = torch.empty((cap, h_kv, d), dtype=torch.float16, device=dev)
        self.v_base_pool = torch.empty((cap, h_kv, s, d4), dtype=torch.uint8, device=dev)
        self.v_res_pool = torch.empty((cap, h_kv, s, d4), dtype=torch.uint8, device=dev)
        self.v_min_pool = torch.empty((cap, h_kv, s), dtype=torch.float16, device=dev)
        self.v_scale_pool = torch.empty((cap, h_kv, s), dtype=torch.float16, device=dev)

    def _pool_names(self) -> tuple[str, ...]:
        """返回全部 8 个物理池张量属性名。"""
        return (
            "k_base_pool",
            "k_res_pool",
            "k_min_pool",
            "k_scale_pool",
            "v_base_pool",
            "v_res_pool",
            "v_min_pool",
            "v_scale_pool",
        )

    def _ensure_free_blocks(self, need: int) -> None:
        """若空闲物理块不足 `need`，倍增扩容物理池并保留已有块索引不变。"""
        avail = len(self.free_blocks)
        if avail >= need:
            return
        old_cap = self.capacity
        new_cap = max(old_cap * 2, old_cap + (need - avail), 16)
        old_tensors = {name: getattr(self, name) for name in self._pool_names()}
        self._alloc_pool_tensors(new_cap)
        for name, old_t in old_tensors.items():
            getattr(self, name)[:old_cap].copy_(old_t)
            del old_t
        self.ref_counts.extend([0] * (new_cap - old_cap))
        # 新扩容块放在空闲栈前端，优先消费原有槽位或按需弹出
        new_ids = list(range(new_cap - 1, old_cap - 1, -1))
        self.free_blocks = new_ids + self.free_blocks
        self.capacity = new_cap

    def _alloc_blocks(self, n: int) -> list[int]:
        """分配 `n` 个物理页块并将各自引用计数置为 1。"""
        if n <= 0:
            return []
        self._ensure_free_blocks(n)
        out: list[int] = []
        for _ in range(n):
            blk = self.free_blocks.pop()
            self.ref_counts[blk] = 1
            out.append(blk)
        return out

    def _release_block(self, blk: int) -> None:
        """递减物理块引用计数，归零时解除前缀哈希绑定并归还空闲列表。"""
        cnt = self.ref_counts[blk]
        if cnt <= 0:
            raise RuntimeError(f"尝试释放引用计数已为 {cnt} 的物理块 {blk}")
        cnt -= 1
        self.ref_counts[blk] = cnt
        if cnt == 0:
            p_key = self._block_prefix_key.pop(blk, None)
            if p_key is not None and self.prefix_table.get(p_key) == blk:
                self.prefix_table.pop(p_key, None)
            self.free_blocks.append(blk)

    def _write_packed_to_blocks(self, blocks: list[int], packed: dict[str, torch.Tensor]) -> None:
        """将打包好的整页切片写入物理池中指定的物理块槽位列表。"""
        if not blocks:
            return
        idx_t = torch.tensor(blocks, dtype=torch.long, device=self.device)
        self.k_base_pool.index_copy_(0, idx_t, packed["k_base"])
        self.k_res_pool.index_copy_(0, idx_t, packed["k_res"])
        self.k_min_pool.index_copy_(0, idx_t, packed["k_min"])
        self.k_scale_pool.index_copy_(0, idx_t, packed["k_scale"])
        self.v_base_pool.index_copy_(0, idx_t, packed["v_base"])
        self.v_res_pool.index_copy_(0, idx_t, packed["v_res"])
        self.v_min_pool.index_copy_(0, idx_t, packed["v_min"])
        self.v_scale_pool.index_copy_(0, idx_t, packed["v_scale"])

    def allocate_sequence(self, seq_id: int | None = None) -> int:
        """注册一条新序列并返回其 `seq_id`。"""
        if seq_id is None:
            while self._next_seq_id in self.sequences:
                self._next_seq_id += 1
            sid = self._next_seq_id
            self._next_seq_id += 1
        else:
            sid = int(seq_id)
            if sid in self.sequences:
                raise ValueError(f"序列 ID {sid} 已存在")
        empty_tail = torch.empty((1, self.num_kv_heads, 0, self.head_dim), dtype=self.kv_dtype, device=self.device)
        self.sequences[sid] = PagedSequenceState(
            seq_id=sid,
            page_size=self.page_size,
            block_ids=[],
            k_tail=empty_tail,
            v_tail=empty_tail.clone(),
        )
        return sid

    def free_sequence(self, seq_id: int) -> None:
        """销毁序列并释放其持有的全部物理页引用与尾部全精缓存。"""
        seq = self.sequences.pop(seq_id, None)
        if seq is None:
            return
        for blk in seq.block_ids:
            self._release_block(blk)
        seq.block_ids.clear()
        seq.k_tail = torch.empty(0, dtype=self.kv_dtype, device=self.device)
        seq.v_tail = torch.empty(0, dtype=self.kv_dtype, device=self.device)
        self._epoch += 1

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int | None = None) -> int:
        """
        Prefix Caching 零拷贝分叉：共享源序列全部已打包物理块并递增引用计数，仅克隆尾部微小全精窗口。
        """
        src = self.sequences[src_seq_id]
        dst_id = self.allocate_sequence(dst_seq_id)
        dst = self.sequences[dst_id]
        dst.block_ids = list(src.block_ids)
        for blk in dst.block_ids:
            self.ref_counts[blk] += 1
        dst.k_tail = src.k_tail.clone()
        dst.v_tail = src.v_tail.clone()
        self._epoch += 1
        return dst_id

    def copy_on_write(self, seq_id: int, logical_page_idx: int) -> int:
        """
        执行 Copy-on-Write (CoW)：当指定逻辑页对应的物理块被多条序列共享 (`ref > 1`) 时，
        分配独占物理块并深拷贝原位面与页元数据，保证其他共享序列绝对不可变。
        """
        seq = self.sequences[seq_id]
        old_blk = seq.block_ids[logical_page_idx]
        if self.ref_counts[old_blk] <= 1:
            return old_blk

        new_blk = self._alloc_blocks(1)[0]
        for name in self._pool_names():
            pool_t = getattr(self, name)
            pool_t[new_blk].copy_(pool_t[old_blk])
        seq.block_ids[logical_page_idx] = new_blk
        self._release_block(old_blk)
        self._epoch += 1
        return new_blk

    def write_page(
        self,
        seq_id: int,
        logical_page_idx: int,
        key_page: torch.Tensor,
        value_page: torch.Tensor,
    ) -> int:
        """先触发 Copy-on-Write，再原地量化打包覆盖指定逻辑页内容。"""
        blk = self.copy_on_write(seq_id, logical_page_idx)
        kp = key_page.view(1, self.num_kv_heads, self.page_size, self.head_dim)
        vp = value_page.view(1, self.num_kv_heads, self.page_size, self.head_dim)
        packed = _quantize_pages_to_packed(kp, vp)
        self._write_packed_to_blocks([blk], packed)
        self._epoch += 1
        return blk

    def append_kv(
        self,
        seq_id: int,
        key_new: torch.Tensor,
        value_new: torch.Tensor,
        recent_tokens: int = 32,
        prefix_keys: list[bytes] | None = None,
    ) -> None:
        """
        向单条序列增量追加 KV，并在满足 `recent_tokens + page_size` 时自动封存新整页并释放全精历史。

        Args:
            seq_id: 目标序列 ID。
            key_new: 形状 `(1, H_kv, L_new, d)`。
            value_new: 形状 `(1, H_kv, L_new, d)`。
            recent_tokens: 末尾保留全精保护的最小 token 数。
            prefix_keys: 可选与每个新打包页一一对应的字节哈希键，用于自动 Prefix Caching 去重。
        """
        seq = self.sequences[seq_id]
        if key_new.shape[0] != 1 or key_new.shape[1] != self.num_kv_heads or key_new.shape[3] != self.head_dim:
            raise ValueError(
                f"append_kv 期望 (1, {self.num_kv_heads}, L, {self.head_dim})，实际收到 {tuple(key_new.shape)}"
            )
        l_new = int(key_new.shape[2])
        if l_new == 0:
            return

        if seq.tail_len == 0:
            k_src, v_src = key_new, value_new
        else:
            k_src = torch.cat([seq.k_tail, key_new], dim=2)
            v_src = torch.cat([seq.v_tail, value_new], dim=2)

        total_avail = int(k_src.shape[2])
        n_pack = max(0, total_avail - recent_tokens) // self.page_size

        if n_pack > 0:
            hist_toks = n_pack * self.page_size
            # 切分出待封存整页，并重排为 (n_pack, H_kv, S, d)
            k_pages = k_src[0, :, :hist_toks, :].view(self.num_kv_heads, n_pack, self.page_size, self.head_dim).permute(1, 0, 2, 3).contiguous()
            v_pages = v_src[0, :, :hist_toks, :].view(self.num_kv_heads, n_pack, self.page_size, self.head_dim).permute(1, 0, 2, 3).contiguous()

            if prefix_keys is not None:
                if len(prefix_keys) != n_pack:
                    raise ValueError(f"prefix_keys 长度 {len(prefix_keys)} 与待封存页数 {n_pack} 不匹配")
                miss_local_idx: list[int] = []
                assigned: list[int] = [-1] * n_pack
                for i, pk in enumerate(prefix_keys):
                    hit_blk = self.prefix_table.get(pk)
                    if hit_blk is not None:
                        self.ref_counts[hit_blk] += 1
                        assigned[i] = hit_blk
                    else:
                        miss_local_idx.append(i)
                if miss_local_idx:
                    idx_tensor = torch.tensor(miss_local_idx, dtype=torch.long, device=self.device)
                    packed_miss = _quantize_pages_to_packed(k_pages.index_select(0, idx_tensor), v_pages.index_select(0, idx_tensor))
                    new_blks = self._alloc_blocks(len(miss_local_idx))
                    self._write_packed_to_blocks(new_blks, packed_miss)
                    for pos_i, new_b in zip(miss_local_idx, new_blks):
                        assigned[pos_i] = new_b
                        pk = prefix_keys[pos_i]
                        self.prefix_table[pk] = new_b
                        self._block_prefix_key[new_b] = pk
                seq.block_ids.extend(assigned)
            else:
                packed = _quantize_pages_to_packed(k_pages, v_pages)
                new_blks = self._alloc_blocks(n_pack)
                self._write_packed_to_blocks(new_blks, packed)
                seq.block_ids.extend(new_blks)

            self._epoch += 1
            # 必须 .contiguous().clone() 创建独立小张量，确保大型 Prefill 输入 Storage 立即释放
            seq.k_tail = k_src[:, :, hist_toks:, :].contiguous().clone()
            seq.v_tail = v_src[:, :, hist_toks:, :].contiguous().clone()
        else:
            # 未触发封页且之前无拼接时同样必须 clone 断开源张量大缓冲区引用
            if seq.tail_len == 0:
                seq.k_tail = k_src.contiguous().clone()
                seq.v_tail = v_src.contiguous().clone()
            else:
                seq.k_tail = k_src
                seq.v_tail = v_src

    def append_batch_kv(
        self,
        seq_ids: list[int],
        key_batch: torch.Tensor,
        value_batch: torch.Tensor,
        recent_tokens: int = 32,
    ) -> None:
        """
        面向齐步并发批次的向量化增量追加入口；当各请求尾部等长时单次量化打包全部新页。
        """
        bsz = len(seq_ids)
        if key_batch.shape[0] != bsz:
            raise ValueError(f"seq_ids 数量 {bsz} 与张量 batch 维 {key_batch.shape[0]} 不一致")
        if bsz == 0 or key_batch.shape[2] == 0:
            return
        if bsz == 1:
            self.append_kv(seq_ids[0], key_batch, value_batch, recent_tokens=recent_tokens)
            return

        seqs = [self.sequences[sid] for sid in seq_ids]
        first_tail = seqs[0].tail_len
        if any(s.tail_len != first_tail for s in seqs):
            for i, sid in enumerate(seq_ids):
                self.append_kv(sid, key_batch[i : i + 1], value_batch[i : i + 1], recent_tokens=recent_tokens)
            return

        if first_tail == 0:
            k_src, v_src = key_batch, value_batch
        else:
            k_prev = torch.cat([s.k_tail for s in seqs], dim=0)
            v_prev = torch.cat([s.v_tail for s in seqs], dim=0)
            k_src = torch.cat([k_prev, key_batch], dim=2)
            v_src = torch.cat([v_prev, value_batch], dim=2)

        total_avail = int(k_src.shape[2])
        n_pack = max(0, total_avail - recent_tokens) // self.page_size

        if n_pack > 0:
            hist_toks = n_pack * self.page_size
            # (B, H_kv, n_pack, S, d) -> (B * n_pack, H_kv, S, d)
            k_pages = (
                k_src[:, :, :hist_toks, :]
                .view(bsz, self.num_kv_heads, n_pack, self.page_size, self.head_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(bsz * n_pack, self.num_kv_heads, self.page_size, self.head_dim)
                .contiguous()
            )
            v_pages = (
                v_src[:, :, :hist_toks, :]
                .view(bsz, self.num_kv_heads, n_pack, self.page_size, self.head_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(bsz * n_pack, self.num_kv_heads, self.page_size, self.head_dim)
                .contiguous()
            )
            packed = _quantize_pages_to_packed(k_pages, v_pages)
            new_blks = self._alloc_blocks(bsz * n_pack)
            self._write_packed_to_blocks(new_blks, packed)
            self._epoch += 1
            for i, seq in enumerate(seqs):
                seq.block_ids.extend(new_blks[i * n_pack : (i + 1) * n_pack])
                seq.k_tail = k_src[i : i + 1, :, hist_toks:, :].contiguous().clone()
                seq.v_tail = v_src[i : i + 1, :, hist_toks:, :].contiguous().clone()
        else:
            for i, seq in enumerate(seqs):
                seq.k_tail = k_src[i : i + 1, :, :, :].contiguous().clone()
                seq.v_tail = v_src[i : i + 1, :, :, :].contiguous().clone()

    def build_batch(self, seq_ids: list[int]) -> PagedKVBatch:
        """
        将任意一组活跃序列（允许不等长历史页与不等长全精尾窗）打包为 `PagedKVBatch` 供 Triton 核消费。
        当物理页映射自上次构建未发生变化时，直接复用缓存的 `block_table` 避免多余 H2D 拷贝。
        """
        if not seq_ids:
            raise ValueError("build_batch 要求至少包含 1 条序列")
        seqs = [self.sequences[sid] for sid in seq_ids]
        bsz = len(seqs)
        cache_key = (self._epoch, tuple(seq_ids))

        if self._bt_cache_key == cache_key and self._bt_cache_val is not None:
            block_table, num_pages_t, num_pages_tup, max_pages, is_uniform = self._bt_cache_val
        else:
            num_pages_list = [s.num_pages for s in seqs]
            num_pages_tup = tuple(num_pages_list)
            max_pages = max(num_pages_list)
            is_uniform = (len(set(num_pages_list)) <= 1)
            block_table = torch.zeros((bsz, max_pages), dtype=torch.int32, device=self.device)
            for i, seq in enumerate(seqs):
                if seq.block_ids:
                    block_table[i, : len(seq.block_ids)] = torch.tensor(seq.block_ids, dtype=torch.int32, device=self.device)
            num_pages_t = torch.tensor(num_pages_list, dtype=torch.int32, device=self.device)
            self._bt_cache_key = cache_key
            self._bt_cache_val = (block_table, num_pages_t, num_pages_tup, max_pages, is_uniform)

        tail_lens_list = [s.tail_len for s in seqs]
        tail_lens_tup = tuple(tail_lens_list)
        max_tail = max(tail_lens_list)

        if max_tail == 0:
            k_tail_b = torch.empty((bsz, self.num_kv_heads, 0, self.head_dim), dtype=self.kv_dtype, device=self.device)
            v_tail_b = torch.empty((bsz, self.num_kv_heads, 0, self.head_dim), dtype=self.kv_dtype, device=self.device)
        elif bsz == 1:
            k_tail_b = seqs[0].k_tail
            v_tail_b = seqs[0].v_tail
        elif all(tl == max_tail for tl in tail_lens_list):
            k_tail_b = torch.cat([s.k_tail for s in seqs], dim=0).contiguous()
            v_tail_b = torch.cat([s.v_tail for s in seqs], dim=0).contiguous()
        else:
            k_tail_b = torch.zeros((bsz, self.num_kv_heads, max_tail, self.head_dim), dtype=self.kv_dtype, device=self.device)
            v_tail_b = torch.zeros((bsz, self.num_kv_heads, max_tail, self.head_dim), dtype=self.kv_dtype, device=self.device)
            for i, seq in enumerate(seqs):
                tl = seq.tail_len
                if tl > 0:
                    k_tail_b[i, :, :tl, :].copy_(seq.k_tail[0])
                    v_tail_b[i, :, :tl, :].copy_(seq.v_tail[0])

        tail_lens_t = torch.tensor(tail_lens_list, dtype=torch.int32, device=self.device)
        seq_lens_t = num_pages_t * self.page_size + tail_lens_t

        return PagedKVBatch(
            pool=self,
            block_table=block_table,
            num_pages_per_seq=num_pages_t,
            k_tail=k_tail_b,
            v_tail=v_tail_b,
            tail_lens=tail_lens_t,
            seq_lens=seq_lens_t,
            num_pages_host=num_pages_tup,
            tail_lens_host=tail_lens_tup,
            max_pages_host=max_pages,
            max_tail_host=max_tail,
            is_uniform_pages=is_uniform,
        )

    def unpack_sequence(self, seq_id: int) -> dict[str, torch.Tensor]:
        """按逻辑页表映射从物理池收集并反量化还原单条序列的完整页级历史 KV。"""
        seq = self.sequences[seq_id]
        n_pages = seq.num_pages
        if n_pages == 0:
            empty = torch.empty((1, self.num_kv_heads, 0, self.head_dim), dtype=torch.float32, device=self.device)
            return {
                "k_base": empty,
                "k_int4": empty,
                "v_base": empty,
                "v_int4": empty,
                "k_tail": seq.k_tail.float(),
                "v_tail": seq.v_tail.float(),
            }

        idx_t = torch.tensor(seq.block_ids, dtype=torch.long, device=self.device)
        kb = self.k_base_pool.index_select(0, idx_t)
        kr = self.k_res_pool.index_select(0, idx_t)
        km = self.k_min_pool.index_select(0, idx_t).float().unsqueeze(2)
        ks = self.k_scale_pool.index_select(0, idx_t).float().unsqueeze(2)

        k_hi = unpack_2bit_quadrants(kb).float()
        k_lo = unpack_2bit_quadrants(kr).float()
        k_base_p = km + (4.0 * k_hi + 1.5) * ks
        k_int4_p = k_base_p + (k_lo - 1.5) * ks

        vb = self.v_base_pool.index_select(0, idx_t)
        vr = self.v_res_pool.index_select(0, idx_t)
        vm = self.v_min_pool.index_select(0, idx_t).float().unsqueeze(-1)
        vs = self.v_scale_pool.index_select(0, idx_t).float().unsqueeze(-1)

        v_hi = unpack_2bit_quadrants(vb).float()
        v_lo = unpack_2bit_quadrants(vr).float()
        v_base_p = vm + (4.0 * v_hi + 1.5) * vs
        v_int4_p = v_base_p + (v_lo - 1.5) * vs

        # (P, H_kv, S, d) -> (1, H_kv, P * S, d)
        hist_len = n_pages * self.page_size
        k_base = k_base_p.permute(1, 0, 2, 3).reshape(1, self.num_kv_heads, hist_len, self.head_dim).contiguous()
        k_int4 = k_int4_p.permute(1, 0, 2, 3).reshape(1, self.num_kv_heads, hist_len, self.head_dim).contiguous()
        v_base = v_base_p.permute(1, 0, 2, 3).reshape(1, self.num_kv_heads, hist_len, self.head_dim).contiguous()
        v_int4 = v_int4_p.permute(1, 0, 2, 3).reshape(1, self.num_kv_heads, hist_len, self.head_dim).contiguous()

        return {
            "k_base": k_base,
            "k_int4": k_int4,
            "v_base": v_base,
            "v_int4": v_int4,
            "k_tail": seq.k_tail.float(),
            "v_tail": seq.v_tail.float(),
        }

    def shuffle_free_blocks(self, seed: int = 42) -> None:
        """打乱当前空闲物理块顺序，用于在基准测试与单测中构造非连续碎片化物理页表。"""
        rng = random.Random(seed)
        rng.shuffle(self.free_blocks)

    def memory_stats(self) -> dict[str, Any]:
        """统计物理池活跃占用、预留占用、Prefix Caching 共享去重收益及单标量有效存储比特。"""
        unique_active_blocks = sum(1 for c in self.ref_counts if c > 0)
        logical_total_blocks = sum(seq.num_pages for seq in self.sequences.values())
        logical_total_tokens = sum(seq.total_seq_len for seq in self.sequences.values())
        tail_bytes = sum(
            int(seq.k_tail.numel() * seq.k_tail.element_size()) + int(seq.v_tail.numel() * seq.v_tail.element_size())
            for seq in self.sequences.values()
        )

        bytes_per_block = (
            2 * self.num_kv_heads * self.page_size * self.d4 * 2  # (k_base+k_res+v_base+v_res)
            + 2 * self.num_kv_heads * self.head_dim * 2          # k_min + k_scale (fp16)
            + 2 * self.num_kv_heads * self.page_size * 2         # v_min + v_scale (fp16)
        )
        physical_pool_active_bytes = unique_active_blocks * bytes_per_block
        physical_pool_reserved_bytes = self.capacity * bytes_per_block
        total_active_bytes = physical_pool_active_bytes + tail_bytes

        logical_kv_elems = 2 * self.num_kv_heads * logical_total_tokens * self.head_dim
        logical_fp16_equiv_bytes = logical_kv_elems * 2

        return {
            "num_sequences": len(self.sequences),
            "unique_active_blocks": unique_active_blocks,
            "logical_total_blocks": logical_total_blocks,
            "prefix_sharing_ratio": float(logical_total_blocks) / max(unique_active_blocks, 1),
            "capacity_blocks": self.capacity,
            "free_blocks": len(self.free_blocks),
            "bytes_per_block": bytes_per_block,
            "physical_pool_active_bytes": physical_pool_active_bytes,
            "physical_pool_reserved_bytes": physical_pool_reserved_bytes,
            "tail_bytes": tail_bytes,
            "total_active_bytes": total_active_bytes,
            "logical_fp16_equiv_bytes": logical_fp16_equiv_bytes,
            "effective_compression_ratio": logical_fp16_equiv_bytes / max(total_active_bytes, 1),
            "physical_bits_per_kv_elem": (total_active_bytes * 8.0) / max(logical_kv_elems, 1),
        }

    @classmethod
    def from_dense_kv(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        page_size: int = 32,
        recent_tokens: int = 32,
        hist_len: int | None = None,
        shuffle_blocks: bool = False,
        seed: int = 42,
    ) -> tuple[PagedBitPlaneKVPool, list[int]]:
        """
        从致密张量 `(B, H_kv, L, d)` 构建真 `PagedBitPlaneKVPool` 及请求 ID 列表的便捷工厂。
        当 `shuffle_blocks=True` 时打乱空闲块栈，刻意制造高度非连续物理页布局以验证真页表寻址。
        """
        if key.shape != value.shape:
            raise ValueError(f"Key 形状 {key.shape} 与 Value 形状 {value.shape} 不一致")
        bsz, h_kv, seq_len, head_dim = key.shape
        if hist_len is None:
            safe_upper = max(0, seq_len - recent_tokens)
            hist_eff = (safe_upper // page_size) * page_size
        else:
            hist_eff = (max(0, min(int(hist_len), seq_len)) // page_size) * page_size
        eff_recent = seq_len - hist_eff
        n_pages_per_seq = hist_eff // page_size
        pool = cls(
            num_kv_heads=h_kv,
            head_dim=head_dim,
            page_size=page_size,
            capacity=max(16, bsz * max(1, n_pages_per_seq) + 8),
            kv_dtype=key.dtype,
            device=key.device,
        )
        if shuffle_blocks:
            pool.shuffle_free_blocks(seed=seed)

        seq_ids = [pool.allocate_sequence() for _ in range(bsz)]
        pool.append_batch_kv(seq_ids, key, value, recent_tokens=eff_recent)
        return pool, seq_ids
