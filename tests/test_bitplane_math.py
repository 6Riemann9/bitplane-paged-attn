"""
Bit-Plane Paged Attention 核心数学恒等式、边界退化、因果零泄密与 GQA 共享单元测试。
"""

from __future__ import annotations

import math

import torch

from bitplane_attn.quant_bitplane import (
    paged_bitplane_attention_decode,
    quantize_nested_4bit_bitplane,
)


def test_nested_bitplane_unpack_identity() -> None:
    """验证高 2 位 Base 与低 2 位 Residual 位面无损拼回原始 4-bit 量化码与反量化值。"""
    torch.manual_seed(0)
    x = torch.randn((4, 8, 64, 128), dtype=torch.float32) * 2.5
    res = quantize_nested_4bit_bitplane(x, reduce_dim=-2)

    q_hi = res["q_hi"]
    q_lo = res["q_lo"]
    x_min = res["x_min"]
    delta_4 = res["delta_4"]

    # 精确位面恒等式校验
    q4_rebuilt = (q_hi << 2) | q_lo
    q4_direct = torch.clamp(torch.round((x - x_min) / delta_4), 0, 15).to(torch.uint8)
    assert torch.equal(q4_rebuilt, q4_direct), "位面拼回结果与直接 4-bit 量化码必须逐元素一致"

    # Base (+1.5) + Residual (q_lo - 1.5) 精确重构校验
    x_rebuilt = res["x_base_hat"] + (q_lo.float() - 1.5) * delta_4
    max_diff = float((x_rebuilt - res["x_int4_hat"]).abs().max().item())
    assert max_diff < 1e-5, f"位面增量重建偏差超限: {max_diff}"


def test_bitplane_boundary_degeneracy() -> None:
    """验证 top_ratio=1.0 时精确退化为 int4，top_ratio=0.0 时精确退化为 base2。"""
    torch.manual_seed(7)
    bsz, h_q, h_kv, lq, lk, head_dim = 1, 8, 2, 1, 256, 128
    q = torch.randn((bsz, h_q, lq, head_dim), dtype=torch.float32)
    k = torch.randn((bsz, h_kv, lk, head_dim), dtype=torch.float32)
    v = torch.randn((bsz, h_kv, lk, head_dim), dtype=torch.float32)
    scaling = 1.0 / math.sqrt(head_dim)

    out_int4, st_int4 = paged_bitplane_attention_decode(
        q, k, v, scaling, mode="int4", page_size=32, recent_tokens=32
    )
    out_bp100, st_bp100 = paged_bitplane_attention_decode(
        q, k, v, scaling, mode="bitplane", page_size=32, recent_tokens=32, top_ratio=1.0
    )
    diff_100 = float((out_int4 - out_bp100).abs().max().item())
    assert diff_100 < 1e-5, f"ratio=1.0 应精确退化为 int4，实际误差={diff_100}"
    assert abs(st_bp100["eff_bits_per_elem"] - 4.0) < 1e-6

    out_base2, _ = paged_bitplane_attention_decode(
        q, k, v, scaling, mode="base2", page_size=32, recent_tokens=32
    )
    out_bp0, st_bp0 = paged_bitplane_attention_decode(
        q, k, v, scaling, mode="bitplane", page_size=32, recent_tokens=32, top_ratio=0.0
    )
    diff_0 = float((out_base2 - out_bp0).abs().max().item())
    assert diff_0 < 1e-5, f"ratio=0.0 应精确退化为 base2，实际误差={diff_0}"
    assert abs(st_bp0["eff_bits_per_elem"] - 2.0) < 1e-6


def test_causal_zero_leakage_in_chunk_eval() -> None:
    """验证多步连续评估 (L_q > 1) 下修改未来查询或 KV 对当前位置 t 输出零影响。"""
    torch.manual_seed(13)
    bsz, h_q, h_kv, past_len, lq, head_dim = 1, 8, 2, 160, 16, 64
    lk = past_len + lq
    scaling = 1.0 / math.sqrt(head_dim)

    q1 = torch.randn((bsz, h_q, lq, head_dim), dtype=torch.float32)
    k1 = torch.randn((bsz, h_kv, lk, head_dim), dtype=torch.float32)
    v1 = torch.randn((bsz, h_kv, lk, head_dim), dtype=torch.float32)

    q2 = q1.clone()
    k2 = k1.clone()
    v2 = v1.clone()

    # 扰动从 t=5 开始的未来查询，以及从 past_len+5 开始的未来 KV
    t_cut = 4
    q2[:, :, t_cut + 1 :, :] += 50.0
    k2[:, :, past_len + t_cut + 1 :, :] -= 80.0
    v2[:, :, past_len + t_cut + 1 :, :] += 80.0

    for mode in ("bitplane", "quest", "int4", "fp16"):
        for routing in ("page_max", "page_mass"):
            if mode != "bitplane" and routing != "page_max":
                continue
            out1, _ = paged_bitplane_attention_decode(
                q1,
                k1,
                v1,
                scaling,
                mode=mode,
                page_size=32,
                recent_tokens=32,
                hist_len=128,
                top_ratio=0.25,
                routing=routing,
                gqa_shared=True,
            )
            out2, _ = paged_bitplane_attention_decode(
                q2,
                k2,
                v2,
                scaling,
                mode=mode,
                page_size=32,
                recent_tokens=32,
                hist_len=128,
                top_ratio=0.25,
                routing=routing,
                gqa_shared=True,
            )
            # out 形状为 (B, L_q, H_q, d)，前 t_cut + 1 步必须完全一致 (0.0 误差)
            prefix_diff = float((out1[:, : t_cut + 1] - out2[:, : t_cut + 1]).abs().max().item())
            assert prefix_diff == 0.0, f"模式 {mode}/{routing} 发生因果未来泄密: diff={prefix_diff}"


def test_gqa_shared_zero_union_inflation() -> None:
    """验证 GQA 组共享选页时并集膨胀倍数严格为 1.0，而每头独立选页在随机 KV 下会出现膨胀。"""
    torch.manual_seed(99)
    bsz, h_q, h_kv, lq, lk, head_dim = 1, 16, 4, 8, 512, 64
    q = torch.randn((bsz, h_q, lq, head_dim), dtype=torch.float32)
    k = torch.randn((bsz, h_kv, lk, head_dim), dtype=torch.float32)
    v = torch.randn((bsz, h_kv, lk, head_dim), dtype=torch.float32)
    scaling = 1.0 / math.sqrt(head_dim)

    _, st_shared = paged_bitplane_attention_decode(
        q, k, v, scaling, mode="bitplane", page_size=32, recent_tokens=32, top_ratio=0.15, gqa_shared=True
    )
    assert abs(st_shared["union_inflation"] - 1.0) < 1e-6
    assert abs(st_shared["union_ratio"] - st_shared["selected_ratio"]) < 1e-6

    _, st_quest_shared = paged_bitplane_attention_decode(
        q, k, v, scaling, mode="quest", page_size=32, recent_tokens=32, top_ratio=0.15, gqa_shared=True
    )
    assert abs(st_quest_shared["union_inflation"] - 1.0) < 1e-6

    _, st_quest_perhead = paged_bitplane_attention_decode(
        q, k, v, scaling, mode="quest", page_size=32, recent_tokens=32, top_ratio=0.15, gqa_shared=False
    )
    assert st_quest_perhead["union_inflation"] > 1.2, (
        f"4:1 GQA 每头独立选页预期产生明显页并集膨胀，实测={st_quest_perhead['union_inflation']}"
    )
