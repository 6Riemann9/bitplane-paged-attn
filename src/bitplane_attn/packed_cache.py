"""
面向 PagedAttention 与 Prefix Caching 的 4-to-1 紧缩物理位面 KV-Cache 模块。

核心设计：
  1. 零洗牌 4 象限紧缩打包（Zero-Shuffle Quadrant 4-to-1 Packing）：
     将头维 d（4 的整数倍，如 64 / 128 / 256）等分为 4 个连续子块 [0:D4], [D4:2D4], [2D4:3D4], [3D4:d]，
     在同一位置 j 上将 4 个 2-bit 整数压入 1 个 uint8 字节（bits [1:0], [3:2], [5:4], [7:6]）。
     在 Triton GPU 核内对 4 个象限均做 100% 连续合并访存，零跨步、零字节内重排洗牌。
  2. 页优先连续内存布局：
     k_base_packed / k_res_packed / v_base_packed / v_res_packed 均为 (B, H_kv, P, page_size, D4) uint8；
     Key 采用页内 per-channel (float16 min/scale, 形状 (B, H_kv, P, d))；
     Value 采用 per-token (float16 min/scale, 形状 (B, H_kv, P, page_size))。
  3. 原生支持尾部全精 KV（未满页及最近局部保护窗口）无缝拼接与精确显存占用统计。
"""

from __future__ import annotations

from dataclasses import dataclass
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
