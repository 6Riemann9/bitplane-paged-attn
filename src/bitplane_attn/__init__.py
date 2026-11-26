"""
Bit-Plane Paged Attention 核心包。

面向混合架构（Hybrid）与 GQA 长上下文大模型的无驱逐、PagedAttention 兼容
嵌套双位面量化（2-bit Base + 按需 Top-k / Single-Pass Anchor-Gated 2-bit Residual）及融合 Triton GPU 解码算子库。
"""

from .hf_cache import (
    BitPlaneCacheLayer,
    BitPlaneDynamicCache,
)
from .packed_cache import (
    PackedBitPlaneKVCache,
    PagedBitPlaneKVPool,
    PagedKVBatch,
    PagedSequenceState,
    pack_2bit_quadrants,
    unpack_2bit_quadrants,
)
from .quant_bitplane import (
    QuantizedBitPlanePages,
    compute_layer_outlier_stats,
    compute_logical_read_bits_per_kv_elem,
    paged_bitplane_attention_decode,
    quantize_nested_4bit_bitplane,
    quantize_uniform_2bit,
)
from .triton_kernels import (
    triton_bitplane_decode,
    triton_bitplane_dualstream_decode,
    triton_bitplane_gated_decode,
    triton_fp16_paged_decode,
    triton_int4_paged_decode,
    triton_paged_bitplane_dualstream_decode,
    triton_paged_int4_decode,
)

__all__ = [
    "BitPlaneCacheLayer",
    "BitPlaneDynamicCache",
    "PackedBitPlaneKVCache",
    "PagedBitPlaneKVPool",
    "PagedKVBatch",
    "PagedSequenceState",
    "QuantizedBitPlanePages",
    "compute_layer_outlier_stats",
    "compute_logical_read_bits_per_kv_elem",
    "pack_2bit_quadrants",
    "paged_bitplane_attention_decode",
    "quantize_nested_4bit_bitplane",
    "quantize_uniform_2bit",
    "triton_bitplane_decode",
    "triton_bitplane_dualstream_decode",
    "triton_bitplane_gated_decode",
    "triton_fp16_paged_decode",
    "triton_int4_paged_decode",
    "triton_paged_bitplane_dualstream_decode",
    "triton_paged_int4_decode",
    "unpack_2bit_quadrants",
]
