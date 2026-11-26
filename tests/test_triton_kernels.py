"""
Triton Split-KV 位面解码核全场景精度与边界单元测试。
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from bitplane_attn.packed_cache import (
    PackedBitPlaneKVCache,
    pack_2bit_quadrants,
    unpack_2bit_quadrants,
)
from bitplane_attn.quant_bitplane import (
    QuantizedBitPlanePages,
    compute_logical_read_bits_per_kv_elem,
    paged_bitplane_attention_decode,
)
from bitplane_attn.triton_kernels import (
    triton_bitplane_decode,
    triton_fp16_paged_decode,
    triton_int4_paged_decode,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU 运行 Triton 核")


def _to_ref_pages(cache: PackedBitPlaneKVCache) -> QuantizedBitPlanePages:
    """将 PackedBitPlaneKVCache 解包至 QuantizedBitPlanePages，用于在完全同源量化网格上校验内核算术零误差。"""
    dec = cache.unpack_to_dense_dequantized()
    k_min_fp32 = cache.k_min.float()
    k_max_fp32 = k_min_fp32 + 15.0 * cache.k_scale.float()
    return QuantizedBitPlanePages(
        page_size=cache.page_size,
        hist_len=cache.hist_len,
        num_pages=cache.num_pages,
        k_base=dec["k_base"],
        k_int4=dec["k_int4"],
        k_int2=dec["k_base"],
        v_base=dec["v_base"],
        v_int4=dec["v_int4"],
        v_int2=dec["v_base"],
        k_page_min=k_min_fp32,
        k_page_max=k_max_fp32,
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_pack_unpack_quadrants_zero_error(head_dim: int) -> None:
    """验证 4 象限 4-to-1 紧缩打包与解包在所有常见头维下逐元素零误差。"""
    torch.manual_seed(42 + head_dim)
    q2 = torch.randint(0, 4, (2, 4, 16, 32, head_dim), dtype=torch.uint8, device="cuda:0")
    packed = pack_2bit_quadrants(q2)
    assert packed.shape == (2, 4, 16, 32, head_dim // 4)
    assert packed.dtype == torch.uint8
    unpacked = unpack_2bit_quadrants(packed)
    assert torch.equal(unpacked, q2), f"d={head_dim} 解包后必须与原 2-bit 张量完全一致"


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("gqa_grp", [1, 2, 4, 8])
@pytest.mark.parametrize("bsz,tail_len", [(1, 0), (1, 17), (4, 32)])
def test_triton_int4_matches_reference(
    head_dim: int,
    gqa_grp: int,
    bsz: int,
    tail_len: int,
) -> None:
    """全面覆盖 d / GQA / B / tail_len 组合，验证 triton_int4_paged_decode 精确对标参考实现。"""
    torch.manual_seed(100 + head_dim + gqa_grp * 7 + bsz * 13 + tail_len)
    h_kv = 2
    h_q = h_kv * gqa_grp
    hist_len = 256  # 8 页 * 32
    lk = hist_len + tail_len
    scaling = 1.0 / math.sqrt(head_dim)

    q = torch.randn((bsz, h_q, 1, head_dim), device="cuda:0", dtype=torch.float16)
    k = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16)
    v = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16)

    cache = PackedBitPlaneKVCache.from_kv(k, v, page_size=32, hist_len=hist_len)
    out_triton = triton_int4_paged_decode(q, cache, scaling)

    # 1) 同源 fp16-scale 网格下验证 Triton 核内核算术严格零误差 (< 2e-3)
    out_ref_same, _ = paged_bitplane_attention_decode(
        q,
        k,
        v,
        scaling,
        mode="int4",
        page_size=32,
        recent_tokens=max(1, tail_len),
        hist_len=hist_len,
        pages_cache=_to_ref_pages(cache),
    )
    max_err_same = float((out_triton.float() - out_ref_same.float()).abs().max().item())
    assert max_err_same < 2e-3, (
        f"INT4 Triton vs 同源参考偏差超限 (d={head_dim}, G={gqa_grp}, B={bsz}, tail={tail_len}): {max_err_same}"
    )

    # 2) 与独立 fp32-scale 量化参考的端到端分布一致性验证（余弦相似度 > 0.998）
    out_ref_fp32, _ = paged_bitplane_attention_decode(
        q, k, v, scaling, mode="int4", page_size=32, recent_tokens=max(1, tail_len), hist_len=hist_len
    )
    cos_sim = float(F.cosine_similarity(out_triton.float().flatten(), out_ref_fp32.float().flatten(), dim=0).item())
    assert cos_sim > 0.998, f"INT4 fp16-scale vs fp32-scale 端到端余弦相似度偏低: {cos_sim}"


@pytest.mark.parametrize("routing", ["page_max", "page_mass"])
@pytest.mark.parametrize("top_ratio", [0.0, 0.15, 1.0])
def test_triton_bitplane_matches_reference_and_boundaries(
    routing: str,
    top_ratio: float,
) -> None:
    """验证 BitPlane 核在边界 ratio=0.0/1.0 及 15%（含离群尖峰针）精确对标参考实现。"""
    torch.manual_seed(2026)
    bsz, h_kv, grp = 2, 4, 4
    h_q = h_kv * grp
    head_dim = 128
    hist_len = 512  # 16 页 * 32
    tail_len = 23
    lk = hist_len + tail_len
    scaling = 1.0 / math.sqrt(head_dim)

    q = torch.randn((bsz, h_q, 1, head_dim), device="cuda:0", dtype=torch.float16)
    k = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.7
    v = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.7

    # 在第 3 页与第 11 页注入强信号针（模拟真实检索页与离群值）
    k[:, :, 3 * 32 + 5, :] = q[:, ::grp, 0, :] * 4.5
    v[:, :, 3 * 32 + 5, :] *= 8.0
    k[:, :, 11 * 32 + 19, :] = q[:, ::grp, 0, :] * 3.8
    v[:, :, 11 * 32 + 19, :] *= 6.0

    cache = PackedBitPlaneKVCache.from_kv(k, v, page_size=32, hist_len=hist_len)
    out_triton, st = triton_bitplane_decode(
        q, cache, scaling, top_ratio=top_ratio, routing=routing, gqa_shared=True, return_stats=True
    )

    if top_ratio == 0.0:
        ref_mode = "base2"
        exp_b = compute_logical_read_bits_per_kv_elem("base2", head_dim=head_dim, page_size=32)
        assert abs(st["logical_read_bits_per_kv_elem"] - exp_b) < 1e-6
    elif top_ratio == 1.0:
        ref_mode = "int4"
        exp_b = compute_logical_read_bits_per_kv_elem("int4", head_dim=head_dim, page_size=32)
        assert abs(st["logical_read_bits_per_kv_elem"] - exp_b) < 1e-6
    else:
        ref_mode = "bitplane"

    out_ref_same, _ = paged_bitplane_attention_decode(
        q,
        k,
        v,
        scaling,
        mode=ref_mode,
        page_size=32,
        recent_tokens=tail_len,
        hist_len=hist_len,
        top_ratio=top_ratio,
        routing=routing,
        gqa_shared=True,
        pages_cache=_to_ref_pages(cache),
    )
    diff = (out_triton.float() - out_ref_same.float()).abs()
    max_err_same = float(diff.max().item())
    rel_l2 = float(diff.norm().item() / max(float(out_ref_same.float().norm().item()), 1e-12))
    scale_tol = max(2e-3, 1.5e-3 * float(out_ref_same.float().abs().max().item()))
    assert rel_l2 < 1e-3 and max_err_same < scale_tol, (
        f"BitPlane Triton vs {ref_mode} 同源网格 (routing={routing}, ratio={top_ratio}) "
        f"rel_l2={rel_l2:.4e}, max_err={max_err_same:.5f}"
    )

    # 验证与全精 SDPA 的余弦一致性：Top-15% 精修必须显著优于纯 2-bit Base (0.927 -> 0.995) 并逼近物理 INT4 上限
    k_exp = k[:, :, None, :, :].expand(bsz, h_kv, grp, lk, head_dim).reshape(bsz, h_q, lk, head_dim)
    v_exp = v[:, :, None, :, :].expand(bsz, h_kv, grp, lk, head_dim).reshape(bsz, h_q, lk, head_dim)
    out_sdpa = F.scaled_dot_product_attention(q.float(), k_exp.float(), v_exp.float(), scale=scaling).transpose(1, 2)
    cos_sim = float(F.cosine_similarity(out_triton.float().flatten(), out_sdpa.float().flatten(), dim=0).item())
    if top_ratio == 0.0:
        assert cos_sim > 0.90
    else:
        assert cos_sim > 0.994, f"BitPlane(ratio={top_ratio}) 相对全精 SDPA 相似度不足: {cos_sim}"


@pytest.mark.parametrize("head_dim,gqa_grp,seq_len", [(128, 4, 533), (256, 4, 1029), (64, 2, 257)])
def test_triton_fp16_matches_pytorch_sdpa(head_dim: int, gqa_grp: int, seq_len: int) -> None:
    """验证 Split-KV 全精 Triton 解码基线与 PyTorch SDPA 精确一致。"""
    torch.manual_seed(77 + head_dim + seq_len)
    bsz, h_kv = 2, 4
    h_q = h_kv * gqa_grp
    scaling = 1.0 / math.sqrt(head_dim)

    q = torch.randn((bsz, h_q, 1, head_dim), device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn((bsz, h_kv, seq_len, head_dim), device="cuda:0", dtype=torch.bfloat16)
    v = torch.randn((bsz, h_kv, seq_len, head_dim), device="cuda:0", dtype=torch.bfloat16)

    out_triton = triton_fp16_paged_decode(q, k, v, scaling)

    k_exp = k[:, :, None, :, :].expand(bsz, h_kv, gqa_grp, seq_len, head_dim).reshape(bsz, h_q, seq_len, head_dim)
    v_exp = v[:, :, None, :, :].expand(bsz, h_kv, gqa_grp, seq_len, head_dim).reshape(bsz, h_q, seq_len, head_dim)
    out_sdpa = F.scaled_dot_product_attention(q.float(), k_exp.float(), v_exp.float(), scale=scaling).transpose(1, 2).contiguous()

    max_err = float((out_triton.float() - out_sdpa.float()).abs().max().item())
    assert max_err < 2e-3, f"FP16 Split-KV Triton vs SDPA 误差={max_err}"



@pytest.mark.parametrize("head_dim,gqa_grp,tail_len", [(64, 2, 0), (128, 4, 23), (256, 4, 17)])
@pytest.mark.parametrize("tau_hot,tau_skip", [(-1e6, 1e6), (1e6, 1e6), (2.0, 4.5), (1.5, 4.0)])
def test_triton_bitplane_gated_matches_reference(
    head_dim: int,
    gqa_grp: int,
    tail_len: int,
    tau_hot: float,
    tau_skip: float,
) -> None:
    """验证 Single-Pass Anchor-Gated Bit-Plane 核覆盖边界与真实稀疏分布，精度与三级分流比均精确对齐参考实现。"""
    from bitplane_attn.triton_kernels import triton_bitplane_gated_decode

    torch.manual_seed(3030 + head_dim + gqa_grp * 11 + tail_len)
    bsz, h_kv = 2, 4
    h_q = h_kv * gqa_grp
    hist_len = 512  # 16 页 * 32
    lk = hist_len + tail_len
    scaling = 1.0 / math.sqrt(head_dim)

    q = torch.randn((bsz, h_q, 1, head_dim), device="cuda:0", dtype=torch.float16)
    k = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.5
    v = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.5

    # Page 0 构造强 Attention Sink，中间注入少许 Hot / Warm 检索峰
    k[:, :, 1, :] = q[:, ::gqa_grp, 0, :] * 3.8
    k[:, :, 5 * 32 + 9, :] = q[:, ::gqa_grp, 0, :] * 3.2
    v[:, :, 5 * 32 + 9, :] *= 4.0
    k[:, :, 12 * 32 + 17, :] = q[:, ::gqa_grp, 0, :] * 2.1

    cache = PackedBitPlaneKVCache.from_kv(k, v, page_size=32, hist_len=hist_len)
    out_triton, st_t = triton_bitplane_gated_decode(
        q, cache, scaling, tau_hot=tau_hot, tau_skip=tau_skip, return_stats=True
    )

    out_ref, st_r = paged_bitplane_attention_decode(
        q,
        k,
        v,
        scaling,
        mode="bitplane_gated",
        page_size=32,
        recent_tokens=max(1, tail_len),
        hist_len=hist_len,
        tau_hot=tau_hot,
        tau_skip=tau_skip,
        pages_cache=_to_ref_pages(cache),
    )

    # 校验三级分流比例严格吻合
    assert abs(st_t["hot_ratio"] - st_r["hot_ratio"]) < 1e-4, f"hot_ratio 不一致: {st_t['hot_ratio']} vs {st_r['hot_ratio']}"
    assert abs(st_t["warm_ratio"] - st_r["warm_ratio"]) < 1e-4, f"warm_ratio 不一致: {st_t['warm_ratio']} vs {st_r['warm_ratio']}"
    assert abs(st_t["skip_ratio"] - st_r["skip_ratio"]) < 1e-4, f"skip_ratio 不一致: {st_t['skip_ratio']} vs {st_r['skip_ratio']}"

    if tau_hot > 1e4 and tau_skip > 1e4:
        assert abs(st_t["hot_ratio"] - 1.0) < 1e-6
        out_i4 = triton_int4_paged_decode(q, cache, scaling)
        assert float((out_triton.float() - out_i4.float()).abs().max().item()) < 2e-3

    diff = (out_triton.float() - out_ref.float()).abs()
    rel_l2 = float(diff.norm().item() / max(float(out_ref.float().norm().item()), 1e-12))
    assert rel_l2 < 1e-3, f"Single-Pass Gated Triton 误差超限: rel_l2={rel_l2:.5e}, max_abs={float(diff.max()):.5f}"


@pytest.mark.parametrize("head_dim,gqa_grp,tail_len", [(64, 2, 0), (128, 4, 23), (256, 4, 17)])
@pytest.mark.parametrize(
    "hot_ratio,warm_ratio,beta",
    [
        (0.15, 0.35, 1.5),
        (0.15, 0.85, 1.5),
        (0.15, 0.0, 1.5),
        (0.0, 0.5, 1.5),
        (1.0, 0.0, 1.5),
        (0.15, 0.35, 0.0),
    ],
)
def test_triton_bitplane_dualstream_matches_reference(
    head_dim: int,
    gqa_grp: int,
    tail_len: int,
    hot_ratio: float,
    warm_ratio: float,
    beta: float,
) -> None:
    """验证 Zero-Branch Ellipsoid-Routed Dual-Stream Triton 核在全部头维/GQA/边界比下严格对齐参考实现。"""
    from bitplane_attn.triton_kernels import triton_bitplane_dualstream_decode

    torch.manual_seed(4040 + head_dim + gqa_grp * 13 + tail_len)
    bsz, h_kv = 2, 4
    h_q = h_kv * gqa_grp
    hist_len = 512  # 16 页 * 32
    lk = hist_len + tail_len
    scaling = 1.0 / math.sqrt(head_dim)

    q = torch.randn((bsz, h_q, 1, head_dim), device="cuda:0", dtype=torch.float16)
    k = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.6
    v = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.6

    # 注入 Page 0 Attention Sink 及跨页检索尖峰
    k[:, :, 2, :] = q[:, ::gqa_grp, 0, :] * 4.2
    k[:, :, 4 * 32 + 11, :] = q[:, ::gqa_grp, 0, :] * 3.6
    v[:, :, 4 * 32 + 11, :] *= 6.0
    k[:, :, 10 * 32 + 21, :] = q[:, ::gqa_grp, 0, :] * 2.5
    v[:, :, 10 * 32 + 21, :] *= 3.0

    cache = PackedBitPlaneKVCache.from_kv(k, v, page_size=32, hist_len=hist_len)
    out_triton, st_t = triton_bitplane_dualstream_decode(
        q,
        cache,
        scaling,
        hot_ratio=hot_ratio,
        warm_ratio=warm_ratio,
        ellipsoid_beta=beta,
        pin_page0=True,
        return_stats=True,
    )

    out_ref, st_r = paged_bitplane_attention_decode(
        q,
        k,
        v,
        scaling,
        mode="bitplane_dualstream",
        page_size=32,
        recent_tokens=max(1, tail_len),
        hist_len=hist_len,
        hot_ratio=hot_ratio,
        warm_ratio=warm_ratio,
        ellipsoid_beta=beta,
        pin_page0=True,
        pages_cache=_to_ref_pages(cache),
    )

    assert abs((st_t["hot_ratio"] + st_t["warm_ratio"] + st_t["skip_ratio"]) - 1.0) < 1e-6
    assert abs(st_t["hot_ratio"] - st_r["hot_ratio"]) < 1e-6
    assert abs(st_t["warm_ratio"] - st_r["warm_ratio"]) < 1e-6

    if hot_ratio >= 1.0:
        out_i4 = triton_int4_paged_decode(q, cache, scaling)
        assert float((out_triton.float() - out_i4.float()).abs().max().item()) < 2e-3

    diff = (out_triton.float() - out_ref.float()).abs()
    rel_l2 = float(diff.norm().item() / max(float(out_ref.float().norm().item()), 1e-12))
    cos_sim = float(F.cosine_similarity(out_triton.float().flatten(), out_ref.float().flatten(), dim=0).item())
    assert cos_sim > 0.999, f"DualStream Triton 余弦一致性不足: cos_sim={cos_sim:.6f}"

    # 对非注入页施加轻微尺度梯度消除均匀噪声并列边界，严格校验双流核内核算术零误差 (< 1e-3)
    k_non_tie = k.clone()
    num_pages = hist_len // 32
    for p_idx in range(num_pages):
        if p_idx not in (0, 4, 10):
            k_non_tie[:, :, p_idx * 32 : (p_idx + 1) * 32, :] *= 1.0 - 0.03 * p_idx
    cache_nt = PackedBitPlaneKVCache.from_kv(k_non_tie, v, page_size=32, hist_len=hist_len)
    out_tri_nt = triton_bitplane_dualstream_decode(
        q,
        cache_nt,
        scaling,
        hot_ratio=hot_ratio,
        warm_ratio=warm_ratio,
        ellipsoid_beta=beta,
        pin_page0=True,
    )
    out_ref_nt, _ = paged_bitplane_attention_decode(
        q,
        k_non_tie,
        v,
        scaling,
        mode="bitplane_dualstream",
        page_size=32,
        recent_tokens=max(1, tail_len),
        hist_len=hist_len,
        hot_ratio=hot_ratio,
        warm_ratio=warm_ratio,
        ellipsoid_beta=beta,
        pin_page0=True,
        pages_cache=_to_ref_pages(cache_nt),
    )
    diff_nt = (out_tri_nt.float() - out_ref_nt.float()).abs()
    rel_l2_nt = float(diff_nt.norm().item() / max(float(out_ref_nt.float().norm().item()), 1e-12))
    assert rel_l2_nt < 1.0e-3, (
        f"DualStream Triton 算术误差超限 (d={head_dim}, G={gqa_grp}, hot={hot_ratio}, warm={warm_ratio}, beta={beta}): "
        f"rel_l2={rel_l2_nt:.5e}, orig_rel_l2={rel_l2:.5e}"
    )
