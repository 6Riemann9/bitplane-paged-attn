"""
面向 Blackwell (RTX 5070 Ti) 的紧缩双位面 Split-KV Triton GPU 解码算子库 (Tensor Core 融合版)。

提供三套采用完全同源 GQA 组共享 + 4 象限连续访存 + Tensor Core + Split-KV FlashDecoding 架构的 GPU 算子：
  1. `triton_bitplane_decode`（主方法）：
     - Stage 1 仅读取 2-bit Base 位面并用仿射预折叠 + FP16 Tensor Core 算出全部页粗分数 (fp16) 与摘要；
     - 融合路由尾核 `_gqa_route_and_tail_kernel` 单核同时完成全精尾部 Window 计算与 GQA 组共享页打分；
     - Stage 2 在同一核内对长尾页 (~85–90%) 零重读 K 且仅读 2-bit V_base，
       对热点页 (~10–15%) 按需读取 K/V Residual 增量精修至 4-bit；
  2. `triton_int4_paged_decode`（强量化基线）：同款 Tensor Core Split-KV 单 Pass 全量读取 4-bit；
  3. `triton_fp16_paged_decode`（全精基线）：同款 Tensor Core Split-KV 读取全精 BF16/FP16 KV。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .packed_cache import PackedBitPlaneKVCache


def _choose_num_splits(batch_size: int, num_kv_heads: int, num_pages: int) -> int:
    """
    自适应选择 Split-KV 并行分片数。
    针对 RTX 5070 Ti (70 SMs) 提高并行分片数以充分隐藏 HBM 访问延迟。
    """
    if num_pages <= 1:
        return max(1, num_pages)
    target_ctas = 512
    total_groups = max(1, batch_size * num_kv_heads)
    ideal_splits = math.ceil(target_ctas / total_groups)
    # 至少保留 ~8 页/分片 以避免过于零碎，上限 64
    max_by_pages = max(1, num_pages // 8)
    # 对中等页数 (< 64 页) 允许更细切分
    if num_pages < 64:
        max_by_pages = max(1, num_pages // 2)
    return max(1, min(64, max_by_pages, ideal_splits))


@triton.jit
def _bitplane_stage1_kernel(
    q_ptr,
    k_base_ptr,
    k_min_ptr,
    k_scale_ptr,
    s_base_out_ptr,
    page_max_ptr,
    page_lse_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb_b,
    stride_kb_h,
    stride_kb_p,
    stride_kb_s,
    stride_kb_d4,
    stride_km_b,
    stride_km_h,
    stride_km_p,
    stride_km_d,
    stride_so_b,
    stride_so_h,
    stride_so_p,
    stride_so_s,
    stride_pm_b,
    stride_pm_h,
    stride_pm_p,
    scaling,
    num_pages,
    num_splits,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    D4: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D4: tl.constexpr,
):
    """Stage 1：全扫 2-bit K_base，仿射预折叠 + FP16 Tensor Core 计算 fp16 s_base 与 fp32 页摘要。"""
    b_kv = tl.program_id(0)
    split_id = tl.program_id(1)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    pages_per_split = tl.cdiv(num_pages, num_splits)
    p_start = split_id * pages_per_split
    p_end = tl.minimum(num_pages, p_start + pages_per_split)
    if p_start >= p_end:
        return

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d4 = tl.arange(0, BLOCK_D4)
    mask_d4 = offs_d4 < D4
    offs_s = tl.arange(0, PAGE_SIZE)

    q_row_ptr = q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh
    q0 = tl.load(q_row_ptr + (0 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_row_ptr + (1 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_row_ptr + (2 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_row_ptr + (3 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling

    kmin_base = k_min_ptr + b * stride_km_b + kv_h * stride_km_h
    kscale_base = k_scale_ptr + b * stride_km_b + kv_h * stride_km_h
    kbase_base = k_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    sout_base = s_base_out_ptr + b * stride_so_b + (kv_h * G + offs_g[:, None]) * stride_so_h
    pm_base = page_max_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h
    pl_base = page_lse_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h

    for p in range(p_start, p_end):
        km_off = p * stride_km_p
        km0 = tl.load(kmin_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km1 = tl.load(kmin_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km2 = tl.load(kmin_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km3 = tl.load(kmin_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        ks0 = tl.load(kscale_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks1 = tl.load(kscale_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks2 = tl.load(kscale_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks3 = tl.load(kscale_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        bias_p = (
            tl.sum(q0 * (km0[None, :] + 1.5 * ks0[None, :]), axis=1)
            + tl.sum(q1 * (km1[None, :] + 1.5 * ks1[None, :]), axis=1)
            + tl.sum(q2 * (km2[None, :] + 1.5 * ks2[None, :]), axis=1)
            + tl.sum(q3 * (km3[None, :] + 1.5 * ks3[None, :]), axis=1)
        )

        kb = tl.load(
            kbase_base + p * stride_kb_p + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4,
            mask=mask_d4[None, :],
            other=0,
        )

        # 使用硬件 FP16 Tensor Core 累加至 FP32，释放 Blackwell 高吞吐算力
        acc_s = tl.broadcast_to(bias_p[:, None], (BLOCK_G, PAGE_SIZE))
        acc_s = tl.dot((q0 * (4.0 * ks0[None, :])).to(tl.float16), tl.trans((kb & 3).to(tl.float16)), acc=acc_s)
        acc_s = tl.dot((q1 * (4.0 * ks1[None, :])).to(tl.float16), tl.trans(((kb >> 2) & 3).to(tl.float16)), acc=acc_s)
        acc_s = tl.dot((q2 * (4.0 * ks2[None, :])).to(tl.float16), tl.trans(((kb >> 4) & 3).to(tl.float16)), acc=acc_s)
        acc_s = tl.dot((q3 * (4.0 * ks3[None, :])).to(tl.float16), tl.trans(((kb >> 6) & 3).to(tl.float16)), acc=acc_s)

        p_max = tl.max(acc_s, axis=1)
        delta_s = acc_s - p_max[:, None]
        # 存相对页最大值的偏移 (fp16)，既将中间表 HBM 带宽砍半，又彻底消除大 logit 下 fp16 截断误差
        tl.store(
            sout_base + p * stride_so_p + offs_s[None, :] * stride_so_s,
            delta_s.to(tl.float16),
            mask=mask_g[:, None],
        )
        p_lse = p_max + tl.log(tl.sum(tl.exp(delta_s), axis=1))
        tl.store(pm_base + p * stride_pm_p, p_max, mask=mask_g)
        tl.store(pl_base + p * stride_pm_p, p_lse, mask=mask_g)


@triton.jit
def _gqa_route_and_tail_kernel(
    q_ptr,
    k_tail_ptr,
    v_tail_ptr,
    page_max_ptr,
    page_lse_ptr,
    tail_m_ptr,
    tail_l_ptr,
    tail_o_ptr,
    grp_scores_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kt_b,
    stride_kt_h,
    stride_kt_s,
    stride_kt_d,
    stride_pm_b,
    stride_pm_h,
    stride_pm_p,
    stride_tm_b,
    stride_tm_h,
    stride_to_b,
    stride_to_h,
    stride_to_d,
    stride_gs_b,
    stride_gs_h,
    stride_gs_p,
    scaling,
    num_pages,
    tail_len,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    COMPUTE_SCORES: tl.constexpr,
    USE_PAGE_MAX: tl.constexpr,
):
    """
    融合尾部全精计算与 GQA 组页得分聚合单核（仅 1 次 launch 替代 12 次 PyTorch 碎操作）：
      1. 若 HAS_TAIL：直接计算组内所有头对尾部窗口的精确 (tail_m, tail_l, tail_o) 与 tail_lse；
      2. 若 COMPUTE_SCORES：计算各头跨页与尾部的全局 global_lse，并汇总写出 GQA 组页得分 grp_scores (B, H_kv, P)。
    """
    b_kv = tl.program_id(0)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    # 1) 尾部窗口在线统计 (L_tail <= 64)
    tail_lse = tl.full((BLOCK_G,), -1e20, tl.float32)
    if HAS_TAIL:
        offs_t = tl.arange(0, BLOCK_T)
        mask_t = offs_t < tail_len
        q_tile = tl.load(
            q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh + offs_d[None, :] * stride_qd,
            mask=mask_g[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32) * scaling
        kt_tile = tl.load(
            k_tail_ptr + b * stride_kt_b + kv_h * stride_kt_h + offs_t[:, None] * stride_kt_s + offs_d[None, :] * stride_kt_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        s_t = tl.dot(q_tile.to(tl.float16), tl.trans(kt_tile.to(tl.float16)))
        s_t = tl.where(mask_t[None, :], s_t, -1e20)

        m_t = tl.max(s_t, axis=1)
        exp_t = tl.where(mask_t[None, :], tl.exp(s_t - m_t[:, None]), 0.0)
        l_t = tl.sum(exp_t, axis=1)
        tail_lse = m_t + tl.log(tl.maximum(l_t, 1e-30))

        w_t = exp_t / tl.maximum(l_t[:, None], 1e-30)
        vt_tile = tl.load(
            v_tail_ptr + b * stride_kt_b + kv_h * stride_kt_h + offs_t[:, None] * stride_kt_s + offs_d[None, :] * stride_kt_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        o_t = tl.dot(w_t.to(tl.float16), vt_tile.to(tl.float16))

        tl.store(tail_m_ptr + b * stride_tm_b + (kv_h * G + offs_g) * stride_tm_h, m_t, mask=mask_g)
        tl.store(tail_l_ptr + b * stride_tm_b + (kv_h * G + offs_g) * stride_tm_h, l_t, mask=mask_g)
        tl.store(
            tail_o_ptr + b * stride_to_b + (kv_h * G + offs_g[:, None]) * stride_to_h + offs_d[None, :] * stride_to_d,
            o_t,
            mask=mask_g[:, None] & mask_d[None, :],
        )

    if not COMPUTE_SCORES:
        return

    # 2) Pass A：对当前 GQA 组求出每头的全局 logsumexp
    pm_base = page_max_ptr + b * stride_pm_b + (kv_h * G + offs_g[:, None]) * stride_pm_h
    pl_base = page_lse_ptr + b * stride_pm_b + (kv_h * G + offs_g[:, None]) * stride_pm_h

    m_acc = tl.where(HAS_TAIL, tail_lse, tl.full((BLOCK_G,), -1e20, tl.float32))
    l_acc = tl.where(HAS_TAIL, tl.full((BLOCK_G,), 1.0, tl.float32), tl.zeros((BLOCK_G,), tl.float32))

    offs_p = tl.arange(0, BLOCK_P)
    for start_p in range(0, num_pages, BLOCK_P):
        p_idx = start_p + offs_p
        mask_p = p_idx < num_pages
        pl_blk = tl.load(pl_base + p_idx[None, :] * stride_pm_p, mask=mask_g[:, None] & mask_p[None, :], other=-1e20)
        m_blk = tl.max(pl_blk, axis=1)
        m_new = tl.maximum(m_acc, m_blk)
        alpha = tl.where(l_acc == 0.0, 0.0, tl.exp(m_acc - m_new))
        e_blk = tl.where(mask_p[None, :], tl.exp(pl_blk - m_new[:, None]), 0.0)
        l_acc = l_acc * alpha + tl.sum(e_blk, axis=1)
        m_acc = m_new

    global_lse = m_acc + tl.log(tl.maximum(l_acc, 1e-30))

    # 3) Pass B：归一化后沿 GQA 组聚合 (max 或 sum) 写出 grp_scores (B, H_kv, P)
    gs_base = grp_scores_ptr + b * stride_gs_b + kv_h * stride_gs_h
    src_base = pm_base if USE_PAGE_MAX else pl_base
    for start_p in range(0, num_pages, BLOCK_P):
        p_idx = start_p + offs_p
        mask_p = p_idx < num_pages
        vals = tl.load(src_base + p_idx[None, :] * stride_pm_p, mask=mask_g[:, None] & mask_p[None, :], other=-1e20)
        probs = tl.where(mask_g[:, None] & mask_p[None, :], tl.exp(vals - global_lse[:, None]), 0.0)
        if USE_PAGE_MAX:
            grp_p = tl.max(probs, axis=0)
        else:
            grp_p = tl.sum(probs, axis=0)
        tl.store(gs_base + p_idx * stride_gs_p, grp_p, mask=mask_p)


@triton.jit
def _bitplane_stage2_kernel(
    q_ptr,
    s_base_ptr,
    page_max_ptr,
    hot_mask_ptr,
    k_res_ptr,
    k_scale_ptr,
    v_base_ptr,
    v_res_ptr,
    v_min_ptr,
    v_scale_ptr,
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_so_b,
    stride_so_h,
    stride_so_p,
    stride_so_s,
    stride_pg_b,
    stride_pg_h,
    stride_pg_p,
    stride_hm_b,
    stride_hm_h,
    stride_hm_p,
    stride_kb_b,
    stride_kb_h,
    stride_kb_p,
    stride_kb_s,
    stride_kb_d4,
    stride_km_b,
    stride_km_h,
    stride_km_p,
    stride_km_d,
    stride_vm_b,
    stride_vm_h,
    stride_vm_p,
    stride_vm_s,
    stride_pm_b,
    stride_pm_h,
    stride_pm_k,
    stride_po_b,
    stride_po_h,
    stride_po_k,
    stride_po_d,
    scaling,
    num_pages,
    num_splits,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    D4: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D4: tl.constexpr,
):
    """
    Stage 2 融合解码核（FP16 Tensor Core 累加至 FP32）：
      - 长尾页 (hot_mask == 0)：读 fp16 s_base + 2-bit V_base 更新 Online Softmax（零重读 K，零读 Residual）；
      - 热点页 (hot_mask == 1)：读 K_res/V_res 增量精修 logits 与 Value 至精确 4-bit。
    """
    b_kv = tl.program_id(0)
    split_id = tl.program_id(1)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    pages_per_split = tl.cdiv(num_pages, num_splits)
    p_start = split_id * pages_per_split
    p_end = tl.minimum(num_pages, p_start + pages_per_split)

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d4 = tl.arange(0, BLOCK_D4)
    mask_d4 = offs_d4 < D4
    offs_s = tl.arange(0, PAGE_SIZE)

    pm_out = part_m_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + split_id * stride_pm_k
    pl_out = part_l_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + split_id * stride_pm_k
    po_out = part_o_ptr + b * stride_po_b + (kv_h * G + offs_g[:, None]) * stride_po_h + split_id * stride_po_k

    if p_start >= p_end:
        tl.store(pm_out, tl.full((BLOCK_G,), -1e20, tl.float32), mask=mask_g)
        tl.store(pl_out, tl.zeros((BLOCK_G,), tl.float32), mask=mask_g)
        z_o = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
        tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        return

    q_row_ptr = q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh
    q0 = tl.load(q_row_ptr + (0 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_row_ptr + (1 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_row_ptr + (2 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_row_ptr + (3 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling

    m_i = tl.full((BLOCK_G,), -1e20, tl.float32)
    l_i = tl.zeros((BLOCK_G,), tl.float32)
    acc0 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc1 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc2 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc3 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)

    s_in_base = s_base_ptr + b * stride_so_b + (kv_h * G + offs_g[:, None]) * stride_so_h
    pmax_in_base = page_max_ptr + b * stride_pg_b + (kv_h * G + offs_g) * stride_pg_h
    hm_base = hot_mask_ptr + b * stride_hm_b + kv_h * stride_hm_h
    kres_base = k_res_ptr + b * stride_kb_b + kv_h * stride_kb_h
    kscale_base = k_scale_ptr + b * stride_km_b + kv_h * stride_km_h
    vbase_base = v_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vres_base = v_res_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vmin_base = v_min_ptr + b * stride_vm_b + kv_h * stride_vm_h
    vscale_base = v_scale_ptr + b * stride_vm_b + kv_h * stride_vm_h

    for p in range(p_start, p_end):
        is_hot = tl.load(hm_base + p * stride_hm_p)
        p_max_val = tl.load(pmax_in_base + p * stride_pg_p, mask=mask_g, other=-1e20)
        delta_s = tl.load(s_in_base + p * stride_so_p + offs_s[None, :] * stride_so_s, mask=mask_g[:, None], other=-1e4).to(tl.float32)
        s_p = p_max_val[:, None] + delta_s

        if is_hot != 0:
            km_off = p * stride_km_p
            ks0 = tl.load(kscale_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
            ks1 = tl.load(kscale_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
            ks2 = tl.load(kscale_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
            ks3 = tl.load(kscale_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

            qks0 = q0 * ks0[None, :]
            qks1 = q1 * ks1[None, :]
            qks2 = q2 * ks2[None, :]
            qks3 = q3 * ks3[None, :]

            res_bias = -1.5 * (
                tl.sum(qks0, axis=1)
                + tl.sum(qks1, axis=1)
                + tl.sum(qks2, axis=1)
                + tl.sum(qks3, axis=1)
            )
            kr = tl.load(
                kres_base + p * stride_kb_p + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4,
                mask=mask_d4[None, :],
                other=0,
            )
            s_p += res_bias[:, None]
            s_p = tl.dot(qks0.to(tl.float16), tl.trans((kr & 3).to(tl.float16)), acc=s_p)
            s_p = tl.dot(qks1.to(tl.float16), tl.trans(((kr >> 2) & 3).to(tl.float16)), acc=s_p)
            s_p = tl.dot(qks2.to(tl.float16), tl.trans(((kr >> 4) & 3).to(tl.float16)), acc=s_p)
            s_p = tl.dot(qks3.to(tl.float16), tl.trans(((kr >> 6) & 3).to(tl.float16)), acc=s_p)

        m_ij = tl.max(s_p, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(l_i == 0.0, 0.0, tl.exp(m_i - m_new))
        p_val = tl.exp(s_p - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p_val, axis=1)
        m_i = m_new

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        acc3 = acc3 * alpha[:, None]

        vm = tl.load(vmin_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
        vs = tl.load(vscale_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
        vb = tl.load(
            vbase_base + p * stride_kb_p + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4,
            mask=mask_d4[None, :],
            other=0,
        )

        if is_hot != 0:
            vr = tl.load(
                vres_base + p * stride_kb_p + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4,
                mask=mask_d4[None, :],
                other=0,
            )
            scalar_bias = tl.sum(p_val * vm[None, :], axis=1)[:, None]
            p_sc = p_val * vs[None, :]
            # 行极大缩放保护：避免极小 p_val * v_scale 下溢 fp16 正常区间
            row_scale = tl.maximum(tl.max(tl.abs(p_sc), axis=1), 1e-12)
            p_sc_n = (p_sc / row_scale[:, None]).to(tl.float16)

            acc0 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, (4.0 * (vb & 3).to(tl.float16) + (vr & 3).to(tl.float16)))
            acc1 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, (4.0 * ((vb >> 2) & 3).to(tl.float16) + ((vr >> 2) & 3).to(tl.float16)))
            acc2 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, (4.0 * ((vb >> 4) & 3).to(tl.float16) + ((vr >> 4) & 3).to(tl.float16)))
            acc3 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, (4.0 * ((vb >> 6) & 3).to(tl.float16) + ((vr >> 6) & 3).to(tl.float16)))
        else:
            scalar_bias = tl.sum(p_val * (vm[None, :] + 1.5 * vs[None, :]), axis=1)[:, None]
            p_sc = 4.0 * p_val * vs[None, :]
            row_scale = tl.maximum(tl.max(tl.abs(p_sc), axis=1), 1e-12)
            p_sc_n = (p_sc / row_scale[:, None]).to(tl.float16)

            acc0 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, (vb & 3).to(tl.float16))
            acc1 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 2) & 3).to(tl.float16))
            acc2 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 4) & 3).to(tl.float16))
            acc3 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 6) & 3).to(tl.float16))

    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(pm_out, m_i, mask=mask_g)
    tl.store(pl_out, l_i, mask=mask_g)
    tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, acc0 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, acc1 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, acc2 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, acc3 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])


@triton.jit
def _int4_splitkv_kernel(
    q_ptr,
    k_base_ptr,
    k_res_ptr,
    k_min_ptr,
    k_scale_ptr,
    v_base_ptr,
    v_res_ptr,
    v_min_ptr,
    v_scale_ptr,
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb_b,
    stride_kb_h,
    stride_kb_p,
    stride_kb_s,
    stride_kb_d4,
    stride_km_b,
    stride_km_h,
    stride_km_p,
    stride_km_d,
    stride_vm_b,
    stride_vm_h,
    stride_vm_p,
    stride_vm_s,
    stride_pm_b,
    stride_pm_h,
    stride_pm_k,
    stride_po_b,
    stride_po_h,
    stride_po_k,
    stride_po_d,
    scaling,
    num_pages,
    num_splits,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    D4: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D4: tl.constexpr,
):
    """强对照基线：单 Pass 全量 4-bit Tensor Core Split-KV 解码核。"""
    b_kv = tl.program_id(0)
    split_id = tl.program_id(1)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    pages_per_split = tl.cdiv(num_pages, num_splits)
    p_start = split_id * pages_per_split
    p_end = tl.minimum(num_pages, p_start + pages_per_split)

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d4 = tl.arange(0, BLOCK_D4)
    mask_d4 = offs_d4 < D4
    offs_s = tl.arange(0, PAGE_SIZE)

    pm_out = part_m_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + split_id * stride_pm_k
    pl_out = part_l_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + split_id * stride_pm_k
    po_out = part_o_ptr + b * stride_po_b + (kv_h * G + offs_g[:, None]) * stride_po_h + split_id * stride_po_k

    if p_start >= p_end:
        tl.store(pm_out, tl.full((BLOCK_G,), -1e20, tl.float32), mask=mask_g)
        tl.store(pl_out, tl.zeros((BLOCK_G,), tl.float32), mask=mask_g)
        z_o = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
        tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        return

    q_row_ptr = q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh
    q0 = tl.load(q_row_ptr + (0 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_row_ptr + (1 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_row_ptr + (2 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_row_ptr + (3 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling

    m_i = tl.full((BLOCK_G,), -1e20, tl.float32)
    l_i = tl.zeros((BLOCK_G,), tl.float32)
    acc0 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc1 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc2 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc3 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)

    kbase_base = k_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    kres_base = k_res_ptr + b * stride_kb_b + kv_h * stride_kb_h
    kmin_base = k_min_ptr + b * stride_km_b + kv_h * stride_km_h
    kscale_base = k_scale_ptr + b * stride_km_b + kv_h * stride_km_h
    vbase_base = v_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vres_base = v_res_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vmin_base = v_min_ptr + b * stride_vm_b + kv_h * stride_vm_h
    vscale_base = v_scale_ptr + b * stride_vm_b + kv_h * stride_vm_h

    for p in range(p_start, p_end):
        km_off = p * stride_km_p
        km0 = tl.load(kmin_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km1 = tl.load(kmin_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km2 = tl.load(kmin_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km3 = tl.load(kmin_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        ks0 = tl.load(kscale_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks1 = tl.load(kscale_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks2 = tl.load(kscale_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks3 = tl.load(kscale_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        bias_k = (
            tl.sum(q0 * km0[None, :], axis=1)
            + tl.sum(q1 * km1[None, :], axis=1)
            + tl.sum(q2 * km2[None, :], axis=1)
            + tl.sum(q3 * km3[None, :], axis=1)
        )

        p_byte_off = p * stride_kb_p + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4
        kb = tl.load(kbase_base + p_byte_off, mask=mask_d4[None, :], other=0)
        kr = tl.load(kres_base + p_byte_off, mask=mask_d4[None, :], other=0)

        s_p = tl.broadcast_to(bias_k[:, None], (BLOCK_G, PAGE_SIZE))
        s_p = tl.dot((q0 * ks0[None, :]).to(tl.float16), tl.trans(4.0 * (kb & 3).to(tl.float16) + (kr & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q1 * ks1[None, :]).to(tl.float16), tl.trans(4.0 * ((kb >> 2) & 3).to(tl.float16) + ((kr >> 2) & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q2 * ks2[None, :]).to(tl.float16), tl.trans(4.0 * ((kb >> 4) & 3).to(tl.float16) + ((kr >> 4) & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q3 * ks3[None, :]).to(tl.float16), tl.trans(4.0 * ((kb >> 6) & 3).to(tl.float16) + ((kr >> 6) & 3).to(tl.float16)), acc=s_p)

        m_ij = tl.max(s_p, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(l_i == 0.0, 0.0, tl.exp(m_i - m_new))
        p_val = tl.exp(s_p - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p_val, axis=1)
        m_i = m_new

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        acc3 = acc3 * alpha[:, None]

        vm = tl.load(vmin_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
        vs = tl.load(vscale_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
        vb = tl.load(vbase_base + p_byte_off, mask=mask_d4[None, :], other=0)
        vr = tl.load(vres_base + p_byte_off, mask=mask_d4[None, :], other=0)

        scalar_bias = tl.sum(p_val * vm[None, :], axis=1)[:, None]
        p_sc = p_val * vs[None, :]
        row_scale = tl.maximum(tl.max(tl.abs(p_sc), axis=1), 1e-12)
        p_sc_n = (p_sc / row_scale[:, None]).to(tl.float16)

        acc0 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * (vb & 3).to(tl.float16) + (vr & 3).to(tl.float16))
        acc1 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 2) & 3).to(tl.float16) + ((vr >> 2) & 3).to(tl.float16))
        acc2 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 4) & 3).to(tl.float16) + ((vr >> 4) & 3).to(tl.float16))
        acc3 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 6) & 3).to(tl.float16) + ((vr >> 6) & 3).to(tl.float16))

    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(pm_out, m_i, mask=mask_g)
    tl.store(pl_out, l_i, mask=mask_g)
    tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, acc0 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, acc1 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, acc2 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, acc3 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])


@triton.jit
def _fp16_splitkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_pm_b,
    stride_pm_h,
    stride_pm_k,
    stride_po_b,
    stride_po_h,
    stride_po_k,
    stride_po_d,
    scaling,
    seq_len,
    num_splits,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """同框架对照基线：Tensor Core Split-KV FlashDecoding 全精 BF16/FP16 解码核。"""
    b_kv = tl.program_id(0)
    split_id = tl.program_id(1)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    total_blocks = tl.cdiv(seq_len, BLOCK_N)
    blocks_per_split = tl.cdiv(total_blocks, num_splits)
    blk_start = split_id * blocks_per_split
    blk_end = tl.minimum(total_blocks, blk_start + blocks_per_split)

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    offs_n = tl.arange(0, BLOCK_N)

    pm_out = part_m_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + split_id * stride_pm_k
    pl_out = part_l_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + split_id * stride_pm_k
    po_out = part_o_ptr + b * stride_po_b + (kv_h * G + offs_g[:, None]) * stride_po_h + split_id * stride_po_k + offs_d[None, :] * stride_po_d

    if blk_start >= blk_end:
        tl.store(pm_out, tl.full((BLOCK_G,), -1e20, tl.float32), mask=mask_g)
        tl.store(pl_out, tl.zeros((BLOCK_G,), tl.float32), mask=mask_g)
        tl.store(po_out, tl.zeros((BLOCK_G, BLOCK_D), tl.float32), mask=mask_g[:, None] & mask_d[None, :])
        return

    q = tl.load(
        q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh + offs_d[None, :] * stride_qd,
        mask=mask_g[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32) * scaling
    q_tc = q.to(tl.float16) if HEAD_DIM < 256 else q

    m_i = tl.full((BLOCK_G,), -1e20, tl.float32)
    l_i = tl.zeros((BLOCK_G,), tl.float32)
    acc = tl.zeros((BLOCK_G, BLOCK_D), tl.float32)

    k_base = k_ptr + b * stride_kb + kv_h * stride_kh
    v_base = v_ptr + b * stride_kb + kv_h * stride_kh

    for blk in range(blk_start, blk_end):
        start_n = blk * BLOCK_N
        pos_n = start_n + offs_n
        mask_n = pos_n < seq_len
        k_tile = tl.load(
            k_base + pos_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        s_p = tl.dot(q.to(k_tile.dtype), tl.trans(k_tile))
        s_p = tl.where(mask_n[None, :], s_p, -1e20)

        m_ij = tl.max(s_p, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(l_i == 0.0, 0.0, tl.exp(m_i - m_new))
        p_val = tl.where(mask_n[None, :], tl.exp(s_p - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_val, axis=1)
        m_i = m_new

        v_tile = tl.load(
            v_base + pos_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p_val.to(v_tile.dtype), v_tile)

    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(pm_out, m_i, mask=mask_g)
    tl.store(pl_out, l_i, mask=mask_g)
    tl.store(po_out, acc * inv_l[:, None], mask=mask_g[:, None] & mask_d[None, :])


@triton.jit
def _splitk_reduce_kernel(
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    tail_m_ptr,
    tail_l_ptr,
    tail_o_ptr,
    out_ptr,
    stride_pm_b,
    stride_pm_h,
    stride_pm_k,
    stride_po_b,
    stride_po_h,
    stride_po_k,
    stride_po_d,
    stride_tm_b,
    stride_tm_h,
    stride_to_b,
    stride_to_h,
    stride_to_d,
    stride_ob,
    stride_oh,
    stride_od,
    num_splits,
    H_q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    """Split-KV 归约核：稳定合并全部页分片与可选尾部全精统计。"""
    bh = tl.program_id(0)
    b = bh // H_q
    h = bh % H_q

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    pm_head = part_m_ptr + b * stride_pm_b + h * stride_pm_h
    pl_head = part_l_ptr + b * stride_pm_b + h * stride_pm_h
    po_head = part_o_ptr + b * stride_po_b + h * stride_po_h

    m_global = tl.full((), -1e20, tl.float32)
    for k in range(0, num_splits):
        l_k = tl.load(pl_head + k * stride_pm_k)
        m_k = tl.load(pm_head + k * stride_pm_k)
        m_global = tl.where(l_k > 0.0, tl.maximum(m_global, m_k), m_global)

    if HAS_TAIL:
        m_t = tl.load(tail_m_ptr + b * stride_tm_b + h * stride_tm_h)
        m_global = tl.maximum(m_global, m_t)

    l_sum = tl.zeros((), tl.float32)
    o_acc = tl.zeros((BLOCK_D,), tl.float32)
    for k in range(0, num_splits):
        l_k = tl.load(pl_head + k * stride_pm_k)
        m_k = tl.load(pm_head + k * stride_pm_k)
        w_k = tl.where(l_k > 0.0, l_k * tl.exp(m_k - m_global), 0.0)
        o_k = tl.load(po_head + k * stride_po_k + offs_d * stride_po_d, mask=mask_d, other=0.0).to(tl.float32)
        l_sum += w_k
        o_acc += w_k * o_k

    if HAS_TAIL:
        l_t = tl.load(tail_l_ptr + b * stride_tm_b + h * stride_tm_h)
        m_t = tl.load(tail_m_ptr + b * stride_tm_b + h * stride_tm_h)
        w_t = l_t * tl.exp(m_t - m_global)
        o_t = tl.load(tail_o_ptr + b * stride_to_b + h * stride_to_h + offs_d * stride_to_d, mask=mask_d, other=0.0).to(tl.float32)
        l_sum += w_t
        o_acc += w_t * o_t

    out_vec = o_acc / tl.maximum(l_sum, 1e-30)
    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od, out_vec, mask=mask_d)


def triton_bitplane_decode(
    query: torch.Tensor,
    cache: PackedBitPlaneKVCache,
    scaling: float,
    top_ratio: float = 0.15,
    routing: str = "page_max",
    gqa_shared: bool = True,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """主方法 Triton 融合核入口：2-bit Base 全扫路由 + Top-k 按需 4-bit 精修 + 2-bit 长尾保底。"""
    if query.shape[2] != 1:
        raise ValueError(f"解码 Triton 核要求 L_q == 1，实际收到 {query.shape[2]}")
    if not gqa_shared:
        raise NotImplementedError("Triton 融合核严格采用零并集膨胀的 GQA 组共享选页路径")

    bsz, h_q, _, head_dim = query.shape
    h_kv = cache.num_kv_heads
    grp = h_q // h_kv
    num_pages = cache.num_pages
    page_size = cache.page_size
    tail_len = cache.tail_len
    has_tail = tail_len > 0

    if num_pages == 0:
        if not has_tail:
            raise ValueError("缓存中既无历史页也无尾部全精 KV")
        k_t = cache.k_tail[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        v_t = cache.v_tail[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        out = F.scaled_dot_product_attention(query.float(), k_t.float(), v_t.float(), scale=scaling).transpose(1, 2).contiguous().to(query.dtype)
        if return_stats:
            return out, {"union_ratio": 0.0, "eff_bits_per_elem": 16.0}
        return out

    d4 = head_dim // 4
    block_g = max(16, triton.next_power_of_2(grp))
    block_d4 = max(16, triton.next_power_of_2(d4))
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_t = max(16, triton.next_power_of_2(max(1, tail_len)))
    num_splits = _choose_num_splits(bsz, h_kv, num_pages)
    device = query.device

    # Stage 1 使用 fp16 写 s_base（较 fp32 HBM 流量直接砍半）
    s_base = torch.empty((bsz, h_q, num_pages, page_size), dtype=torch.float16, device=device)
    page_max = torch.empty((bsz, h_q, num_pages), dtype=torch.float32, device=device)
    page_lse = torch.empty((bsz, h_q, num_pages), dtype=torch.float32, device=device)

    q_c = query.squeeze(2).contiguous()
    grid_kv = (bsz * h_kv, num_splits)
    _bitplane_stage1_kernel[grid_kv](
        q_c,
        cache.k_base_packed,
        cache.k_min,
        cache.k_scale,
        s_base,
        page_max,
        page_lse,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        cache.k_base_packed.stride(0),
        cache.k_base_packed.stride(1),
        cache.k_base_packed.stride(2),
        cache.k_base_packed.stride(3),
        cache.k_base_packed.stride(4),
        cache.k_min.stride(0),
        cache.k_min.stride(1),
        cache.k_min.stride(2),
        cache.k_min.stride(3),
        s_base.stride(0),
        s_base.stride(1),
        s_base.stride(2),
        s_base.stride(3),
        page_max.stride(0),
        page_max.stride(1),
        page_max.stride(2),
        scaling,
        num_pages,
        num_splits,
        H_kv=h_kv,
        G=grp,
        D4=d4,
        PAGE_SIZE=page_size,
        BLOCK_G=block_g,
        BLOCK_D4=block_d4,
        num_warps=4,
    )

    # 融合尾部全精计算 + GQA 组页打分
    if top_ratio <= 0.0:
        top_k = 0
        need_scores = False
        use_page_max = True
        hot_mask = torch.zeros((bsz, h_kv, num_pages), dtype=torch.uint8, device=device)
    elif top_ratio >= 1.0:
        top_k = num_pages
        need_scores = False
        use_page_max = True
        hot_mask = torch.ones((bsz, h_kv, num_pages), dtype=torch.uint8, device=device)
    else:
        top_k = max(1, min(num_pages, int(math.ceil(num_pages * top_ratio))))
        need_scores = True
        if routing == "page_max":
            use_page_max = True
        elif routing == "page_mass":
            use_page_max = False
        else:
            raise ValueError(f"不支持的 routing 策略: {routing}")
        hot_mask = torch.zeros((bsz, h_kv, num_pages), dtype=torch.uint8, device=device)

    tail_m = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    tail_l = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    tail_o = torch.empty((bsz, h_q, head_dim), dtype=torch.float32, device=device)
    grp_scores = torch.empty((bsz, h_kv, num_pages), dtype=torch.float32, device=device)

    if has_tail or need_scores:
        # 若尾部 > 64 (极少见)，退回安全 PyTorch 尾巴统计并跳过核内 tail；正常情况全走融合核
        if tail_len <= 64:
            kt_ref = cache.k_tail if has_tail else q_c[:, :h_kv, :1]
            vt_ref = cache.v_tail if has_tail else q_c[:, :h_kv, :1]
            _gqa_route_and_tail_kernel[(bsz * h_kv,)](
                q_c,
                kt_ref,
                vt_ref,
                page_max,
                page_lse,
                tail_m,
                tail_l,
                tail_o,
                grp_scores,
                q_c.stride(0),
                q_c.stride(1),
                q_c.stride(2),
                kt_ref.stride(0),
                kt_ref.stride(1),
                kt_ref.stride(2) if kt_ref.ndim == 4 else 1,
                kt_ref.stride(3) if kt_ref.ndim == 4 else 1,
                page_max.stride(0),
                page_max.stride(1),
                page_max.stride(2),
                tail_m.stride(0),
                tail_m.stride(1),
                tail_o.stride(0),
                tail_o.stride(1),
                tail_o.stride(2),
                grp_scores.stride(0),
                grp_scores.stride(1),
                grp_scores.stride(2),
                scaling,
                num_pages,
                tail_len,
                H_kv=h_kv,
                G=grp,
                HEAD_DIM=head_dim,
                BLOCK_G=block_g,
                BLOCK_D=block_d,
                BLOCK_T=block_t,
                BLOCK_P=128,
                HAS_TAIL=has_tail,
                COMPUTE_SCORES=need_scores,
                USE_PAGE_MAX=use_page_max,
                num_warps=4,
            )
        else:
            q_g = query.squeeze(2).view(bsz, h_kv, grp, head_dim).float()
            scores_t = torch.matmul(q_g, cache.k_tail.float().transpose(-2, -1)) * scaling
            tm_g = scores_t.max(dim=-1).values
            exp_t = torch.exp(scores_t - tm_g.unsqueeze(-1))
            tl_g = exp_t.sum(dim=-1)
            to_g = torch.matmul(exp_t / torch.clamp(tl_g.unsqueeze(-1), min=1e-30), cache.v_tail.float())
            tail_m = tm_g.reshape(bsz, h_q)
            tail_l = tl_g.reshape(bsz, h_q)
            tail_o = to_g.reshape(bsz, h_q, head_dim)
            if need_scores:
                tail_lse = tail_m + torch.log(torch.clamp(tail_l, min=1e-30))
                glse = torch.logsumexp(torch.cat([page_lse, tail_lse.unsqueeze(-1)], dim=-1), dim=-1, keepdim=True)
                if use_page_max:
                    grp_scores = torch.exp(page_max - glse).view(bsz, h_kv, grp, num_pages).max(dim=2).values
                else:
                    grp_scores = torch.exp(page_lse - glse).view(bsz, h_kv, grp, num_pages).sum(dim=2)

    if need_scores:
        topk_idx = torch.topk(grp_scores, k=top_k, dim=-1).indices
        hot_mask.scatter_(-1, topk_idx, 1)

    part_m = torch.empty((bsz, h_q, num_splits), dtype=torch.float32, device=device)
    part_l = torch.empty((bsz, h_q, num_splits), dtype=torch.float32, device=device)
    part_o = torch.empty((bsz, h_q, num_splits, head_dim), dtype=torch.float32, device=device)

    _bitplane_stage2_kernel[grid_kv](
        q_c,
        s_base,
        page_max,
        hot_mask,
        cache.k_res_packed,
        cache.k_scale,
        cache.v_base_packed,
        cache.v_res_packed,
        cache.v_min,
        cache.v_scale,
        part_m,
        part_l,
        part_o,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        s_base.stride(0),
        s_base.stride(1),
        s_base.stride(2),
        s_base.stride(3),
        page_max.stride(0),
        page_max.stride(1),
        page_max.stride(2),
        hot_mask.stride(0),
        hot_mask.stride(1),
        hot_mask.stride(2),
        cache.k_res_packed.stride(0),
        cache.k_res_packed.stride(1),
        cache.k_res_packed.stride(2),
        cache.k_res_packed.stride(3),
        cache.k_res_packed.stride(4),
        cache.k_scale.stride(0),
        cache.k_scale.stride(1),
        cache.k_scale.stride(2),
        cache.k_scale.stride(3),
        cache.v_min.stride(0),
        cache.v_min.stride(1),
        cache.v_min.stride(2),
        cache.v_min.stride(3),
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        scaling,
        num_pages,
        num_splits,
        H_kv=h_kv,
        G=grp,
        D4=d4,
        PAGE_SIZE=page_size,
        BLOCK_G=block_g,
        BLOCK_D4=block_d4,
        num_warps=4,
    )

    out_fp32 = torch.empty((bsz, 1, h_q, head_dim), dtype=torch.float32, device=device)
    out_v = out_fp32.squeeze(1)
    _splitk_reduce_kernel[(bsz * h_q,)](
        part_m,
        part_l,
        part_o,
        tail_m,
        tail_l,
        tail_o,
        out_v,
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        tail_m.stride(0),
        tail_m.stride(1),
        tail_o.stride(0),
        tail_o.stride(1),
        tail_o.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        num_splits,
        H_q=h_q,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        HAS_TAIL=has_tail,
    )
    out = out_fp32.to(query.dtype)

    if return_stats:
        actual_ratio = top_k / num_pages
        return out, {
            "union_ratio": actual_ratio,
            "union_inflation": 1.0,
            "eff_bits_per_elem": 2.0 + 2.0 * actual_ratio,
            "num_splits": num_splits,
        }
    return out


def triton_int4_paged_decode(
    query: torch.Tensor,
    cache: PackedBitPlaneKVCache,
    scaling: float,
) -> torch.Tensor:
    """全量 4-bit Tensor Core Split-KV Triton 解码基线。"""
    if query.shape[2] != 1:
        raise ValueError(f"解码 Triton 核要求 L_q == 1，实际收到 {query.shape[2]}")

    bsz, h_q, _, head_dim = query.shape
    h_kv = cache.num_kv_heads
    grp = h_q // h_kv
    num_pages = cache.num_pages
    page_size = cache.page_size
    tail_len = cache.tail_len
    has_tail = tail_len > 0

    if num_pages == 0:
        k_t = cache.k_tail[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        v_t = cache.v_tail[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        return F.scaled_dot_product_attention(query.float(), k_t.float(), v_t.float(), scale=scaling).transpose(1, 2).contiguous().to(query.dtype)

    d4 = head_dim // 4
    block_g = max(16, triton.next_power_of_2(grp))
    block_d4 = max(16, triton.next_power_of_2(d4))
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_t = max(16, triton.next_power_of_2(max(1, tail_len)))
    num_splits = _choose_num_splits(bsz, h_kv, num_pages)
    device = query.device

    q_c = query.squeeze(2).contiguous()
    part_m = torch.empty((bsz, h_q, num_splits), dtype=torch.float32, device=device)
    part_l = torch.empty((bsz, h_q, num_splits), dtype=torch.float32, device=device)
    part_o = torch.empty((bsz, h_q, num_splits, head_dim), dtype=torch.float32, device=device)

    _int4_splitkv_kernel[(bsz * h_kv, num_splits)](
        q_c,
        cache.k_base_packed,
        cache.k_res_packed,
        cache.k_min,
        cache.k_scale,
        cache.v_base_packed,
        cache.v_res_packed,
        cache.v_min,
        cache.v_scale,
        part_m,
        part_l,
        part_o,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        cache.k_base_packed.stride(0),
        cache.k_base_packed.stride(1),
        cache.k_base_packed.stride(2),
        cache.k_base_packed.stride(3),
        cache.k_base_packed.stride(4),
        cache.k_min.stride(0),
        cache.k_min.stride(1),
        cache.k_min.stride(2),
        cache.k_min.stride(3),
        cache.v_min.stride(0),
        cache.v_min.stride(1),
        cache.v_min.stride(2),
        cache.v_min.stride(3),
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        scaling,
        num_pages,
        num_splits,
        H_kv=h_kv,
        G=grp,
        D4=d4,
        PAGE_SIZE=page_size,
        BLOCK_G=block_g,
        BLOCK_D4=block_d4,
        num_warps=4,
    )

    tail_m = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    tail_l = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    tail_o = torch.empty((bsz, h_q, head_dim), dtype=torch.float32, device=device)
    if has_tail:
        if tail_len <= 64:
            dummy_p = torch.empty((bsz, h_q, 1), dtype=torch.float32, device=device)
            dummy_gs = torch.empty((bsz, h_kv, 1), dtype=torch.float32, device=device)
            _gqa_route_and_tail_kernel[(bsz * h_kv,)](
                q_c,
                cache.k_tail,
                cache.v_tail,
                dummy_p,
                dummy_p,
                tail_m,
                tail_l,
                tail_o,
                dummy_gs,
                q_c.stride(0),
                q_c.stride(1),
                q_c.stride(2),
                cache.k_tail.stride(0),
                cache.k_tail.stride(1),
                cache.k_tail.stride(2),
                cache.k_tail.stride(3),
                dummy_p.stride(0),
                dummy_p.stride(1),
                dummy_p.stride(2),
                tail_m.stride(0),
                tail_m.stride(1),
                tail_o.stride(0),
                tail_o.stride(1),
                tail_o.stride(2),
                dummy_gs.stride(0),
                dummy_gs.stride(1),
                dummy_gs.stride(2),
                scaling,
                1,
                tail_len,
                H_kv=h_kv,
                G=grp,
                HEAD_DIM=head_dim,
                BLOCK_G=block_g,
                BLOCK_D=block_d,
                BLOCK_T=block_t,
                BLOCK_P=16,
                HAS_TAIL=True,
                COMPUTE_SCORES=False,
                USE_PAGE_MAX=True,
                num_warps=4,
            )
        else:
            q_g = query.squeeze(2).view(bsz, h_kv, grp, head_dim).float()
            scores_t = torch.matmul(q_g, cache.k_tail.float().transpose(-2, -1)) * scaling
            tm_g = scores_t.max(dim=-1).values
            exp_t = torch.exp(scores_t - tm_g.unsqueeze(-1))
            tl_g = exp_t.sum(dim=-1)
            to_g = torch.matmul(exp_t / torch.clamp(tl_g.unsqueeze(-1), min=1e-30), cache.v_tail.float())
            tail_m = tm_g.reshape(bsz, h_q)
            tail_l = tl_g.reshape(bsz, h_q)
            tail_o = to_g.reshape(bsz, h_q, head_dim)

    out_fp32 = torch.empty((bsz, 1, h_q, head_dim), dtype=torch.float32, device=device)
    out_v = out_fp32.squeeze(1)
    _splitk_reduce_kernel[(bsz * h_q,)](
        part_m,
        part_l,
        part_o,
        tail_m,
        tail_l,
        tail_o,
        out_v,
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        tail_m.stride(0),
        tail_m.stride(1),
        tail_o.stride(0),
        tail_o.stride(1),
        tail_o.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        num_splits,
        H_q=h_q,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        HAS_TAIL=has_tail,
    )
    return out_fp32.to(query.dtype)


def triton_fp16_paged_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    block_n: int = 32,
) -> torch.Tensor:
    """全精 BF16/FP16 Split-KV Tensor Core Triton 解码基线。"""
    if query.shape[2] != 1:
        raise ValueError(f"解码 Triton 核要求 L_q == 1，实际收到 {query.shape[2]}")

    bsz, h_q, _, head_dim = query.shape
    _, h_kv, seq_len, _ = key.shape
    grp = h_q // h_kv
    block_n_eff = 16 if head_dim >= 256 else block_n
    num_stages_eff = 1 if head_dim >= 256 else 2
    num_blocks = math.ceil(seq_len / block_n_eff)
    num_splits = _choose_num_splits(bsz, h_kv, num_blocks)

    block_g = max(16, triton.next_power_of_2(grp))
    block_d = max(16, triton.next_power_of_2(head_dim))
    device = query.device

    q_c = query.squeeze(2).contiguous()
    part_m = torch.empty((bsz, h_q, num_splits), dtype=torch.float32, device=device)
    part_l = torch.empty((bsz, h_q, num_splits), dtype=torch.float32, device=device)
    part_o = torch.empty((bsz, h_q, num_splits, head_dim), dtype=torch.float32, device=device)

    _fp16_splitkv_kernel[(bsz * h_kv, num_splits)](
        q_c,
        key,
        value,
        part_m,
        part_l,
        part_o,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        scaling,
        seq_len,
        num_splits,
        H_kv=h_kv,
        G=grp,
        HEAD_DIM=head_dim,
        BLOCK_G=block_g,
        BLOCK_N=block_n_eff,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=num_stages_eff,
    )

    dummy_m = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    dummy_l = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    dummy_o = torch.empty((bsz, h_q, head_dim), dtype=torch.float32, device=device)
    out_fp32 = torch.empty((bsz, 1, h_q, head_dim), dtype=torch.float32, device=device)
    out_v = out_fp32.squeeze(1)

    _splitk_reduce_kernel[(bsz * h_q,)](
        part_m,
        part_l,
        part_o,
        dummy_m,
        dummy_l,
        dummy_o,
        out_v,
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        dummy_m.stride(0),
        dummy_m.stride(1),
        dummy_o.stride(0),
        dummy_o.stride(1),
        dummy_o.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        num_splits,
        H_q=h_q,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        HAS_TAIL=False,
    )
    return out_fp32.to(query.dtype)



@triton.jit
def _anchor_and_tail_kernel(
    q_ptr,
    k_base_ptr,
    k_min_ptr,
    k_scale_ptr,
    k_tail_ptr,
    v_tail_ptr,
    tail_m_ptr,
    tail_l_ptr,
    tail_o_ptr,
    m_anchor_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb_b,
    stride_kb_h,
    stride_kb_s,
    stride_kb_d4,
    stride_km_b,
    stride_km_h,
    stride_km_d,
    stride_kt_b,
    stride_kt_h,
    stride_kt_s,
    stride_kt_d,
    stride_tm_b,
    stride_tm_h,
    stride_to_b,
    stride_to_h,
    stride_to_d,
    scaling,
    tail_len,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D4: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D4: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    COMPUTE_TAIL: tl.constexpr,
):
    """
    锚点与尾部融合单核（Grid = B * H_kv）：
      1. 可选计算尾部全精小窗口的 Online Softmax 统计 (tail_m, tail_l, tail_o)；
      2. 读取 Page 0 (Attention Sink) 2-bit K_base 计算 p0_max，与 tail_m 取逐头最大得出 m_anchor。
    """
    b_kv = tl.program_id(0)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    offs_d4 = tl.arange(0, BLOCK_D4)
    mask_d4 = offs_d4 < D4
    offs_s = tl.arange(0, PAGE_SIZE)

    m_t = tl.full((BLOCK_G,), -1e20, tl.float32)
    if HAS_TAIL and COMPUTE_TAIL:
        offs_t = tl.arange(0, BLOCK_T)
        mask_t = offs_t < tail_len
        q_tile = tl.load(
            q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh + offs_d[None, :] * stride_qd,
            mask=mask_g[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32) * scaling
        kt_tile = tl.load(
            k_tail_ptr + b * stride_kt_b + kv_h * stride_kt_h + offs_t[:, None] * stride_kt_s + offs_d[None, :] * stride_kt_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        s_t = tl.dot(q_tile.to(tl.float16), tl.trans(kt_tile.to(tl.float16)))
        s_t = tl.where(mask_t[None, :], s_t, -1e20)

        m_t = tl.max(s_t, axis=1)
        exp_t = tl.where(mask_t[None, :], tl.exp(s_t - m_t[:, None]), 0.0)
        l_t = tl.sum(exp_t, axis=1)
        w_t = exp_t / tl.maximum(l_t[:, None], 1e-30)
        vt_tile = tl.load(
            v_tail_ptr + b * stride_kt_b + kv_h * stride_kt_h + offs_t[:, None] * stride_kt_s + offs_d[None, :] * stride_kt_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        o_t = tl.dot(w_t.to(tl.float16), vt_tile.to(tl.float16))

        tl.store(tail_m_ptr + b * stride_tm_b + (kv_h * G + offs_g) * stride_tm_h, m_t, mask=mask_g)
        tl.store(tail_l_ptr + b * stride_tm_b + (kv_h * G + offs_g) * stride_tm_h, l_t, mask=mask_g)
        tl.store(
            tail_o_ptr + b * stride_to_b + (kv_h * G + offs_g[:, None]) * stride_to_h + offs_d[None, :] * stride_to_d,
            o_t,
            mask=mask_g[:, None] & mask_d[None, :],
        )

    # 计算 Page 0 (Attention Sink) 的 2-bit K_base 粗分最大值 p0_max
    q_row_ptr = q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh
    q0 = tl.load(q_row_ptr + (0 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_row_ptr + (1 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_row_ptr + (2 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_row_ptr + (3 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling

    kmin_p0 = k_min_ptr + b * stride_km_b + kv_h * stride_km_h
    kscale_p0 = k_scale_ptr + b * stride_km_b + kv_h * stride_km_h
    km0 = tl.load(kmin_p0 + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
    km1 = tl.load(kmin_p0 + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
    km2 = tl.load(kmin_p0 + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
    km3 = tl.load(kmin_p0 + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
    ks0 = tl.load(kscale_p0 + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
    ks1 = tl.load(kscale_p0 + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
    ks2 = tl.load(kscale_p0 + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
    ks3 = tl.load(kscale_p0 + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

    bias_p0 = (
        tl.sum(q0 * (km0[None, :] + 1.5 * ks0[None, :]), axis=1)
        + tl.sum(q1 * (km1[None, :] + 1.5 * ks1[None, :]), axis=1)
        + tl.sum(q2 * (km2[None, :] + 1.5 * ks2[None, :]), axis=1)
        + tl.sum(q3 * (km3[None, :] + 1.5 * ks3[None, :]), axis=1)
    )
    kb0 = tl.load(
        k_base_ptr + b * stride_kb_b + kv_h * stride_kb_h + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4,
        mask=mask_d4[None, :],
        other=0,
    )
    acc_p0 = tl.broadcast_to(bias_p0[:, None], (BLOCK_G, PAGE_SIZE))
    acc_p0 = tl.dot((q0 * (4.0 * ks0[None, :])).to(tl.float16), tl.trans((kb0 & 3).to(tl.float16)), acc=acc_p0)
    acc_p0 = tl.dot((q1 * (4.0 * ks1[None, :])).to(tl.float16), tl.trans(((kb0 >> 2) & 3).to(tl.float16)), acc=acc_p0)
    acc_p0 = tl.dot((q2 * (4.0 * ks2[None, :])).to(tl.float16), tl.trans(((kb0 >> 4) & 3).to(tl.float16)), acc=acc_p0)
    acc_p0 = tl.dot((q3 * (4.0 * ks3[None, :])).to(tl.float16), tl.trans(((kb0 >> 6) & 3).to(tl.float16)), acc=acc_p0)

    p0_max = tl.max(acc_p0, axis=1)
    m_anchor = tl.maximum(m_t, p0_max) if (HAS_TAIL and COMPUTE_TAIL) else p0_max
    tl.store(m_anchor_ptr + b * stride_tm_b + (kv_h * G + offs_g) * stride_tm_h, m_anchor, mask=mask_g)


@triton.jit
def _bitplane_singlepass_gated_kernel(
    q_ptr,
    m_anchor_ptr,
    k_base_ptr,
    k_res_ptr,
    k_min_ptr,
    k_scale_ptr,
    v_base_ptr,
    v_res_ptr,
    v_min_ptr,
    v_scale_ptr,
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    tier_out_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_ma_b,
    stride_ma_h,
    stride_kb_b,
    stride_kb_h,
    stride_kb_p,
    stride_kb_s,
    stride_kb_d4,
    stride_km_b,
    stride_km_h,
    stride_km_p,
    stride_km_d,
    stride_vm_b,
    stride_vm_h,
    stride_vm_p,
    stride_vm_s,
    stride_pm_b,
    stride_pm_h,
    stride_pm_k,
    stride_po_b,
    stride_po_h,
    stride_po_k,
    stride_po_d,
    stride_to_b,
    stride_to_h,
    stride_to_p,
    scaling,
    tau_hot,
    tau_skip,
    num_pages,
    num_splits,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    D4: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D4: tl.constexpr,
    RECORD_TIERS: tl.constexpr,
):
    """
    Single-Pass Anchor-Gated Bit-Plane 核心解码核（零中间分数表、零 Top-k、零重读 k_scale）：
      - 仅扫描 2-bit K_base 并与静态全局锚点比对得出组共享三级路径：
        * Skip (<= -tau_skip 且 p != 0): 彻底跳过 k_res、Softmax 与全部 Value 读入；
        * Warm (-tau_skip < delta <= -tau_hot 且 p != 0): 复用 K_base 分数，仅读 2-bit V_base；
        * Hot (> -tau_hot 或 p == 0): 复用寄存器 K_scale + 按需读 K_res/V_res 精修至 4-bit。
    """
    b_kv = tl.program_id(0)
    split_id = tl.program_id(1)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    pages_per_split = tl.cdiv(num_pages, num_splits)
    p_start = split_id * pages_per_split
    p_end = tl.minimum(num_pages, p_start + pages_per_split)

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d4 = tl.arange(0, BLOCK_D4)
    mask_d4 = offs_d4 < D4
    offs_s = tl.arange(0, PAGE_SIZE)

    pm_out = part_m_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + split_id * stride_pm_k
    pl_out = part_l_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + split_id * stride_pm_k
    po_out = part_o_ptr + b * stride_po_b + (kv_h * G + offs_g[:, None]) * stride_po_h + split_id * stride_po_k

    if p_start >= p_end:
        tl.store(pm_out, tl.full((BLOCK_G,), -1e20, tl.float32), mask=mask_g)
        tl.store(pl_out, tl.zeros((BLOCK_G,), tl.float32), mask=mask_g)
        z_o = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
        tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        return

    q_row_ptr = q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh
    q0 = tl.load(q_row_ptr + (0 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_row_ptr + (1 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_row_ptr + (2 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_row_ptr + (3 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling

    m_anchor = tl.load(m_anchor_ptr + b * stride_ma_b + (kv_h * G + offs_g) * stride_ma_h, mask=mask_g, other=-1e20)

    m_i = tl.full((BLOCK_G,), -1e20, tl.float32)
    l_i = tl.zeros((BLOCK_G,), tl.float32)
    acc0 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc1 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc2 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc3 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)

    kbase_base = k_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    kres_base = k_res_ptr + b * stride_kb_b + kv_h * stride_kb_h
    kmin_base = k_min_ptr + b * stride_km_b + kv_h * stride_km_h
    kscale_base = k_scale_ptr + b * stride_km_b + kv_h * stride_km_h
    vbase_base = v_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vres_base = v_res_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vmin_base = v_min_ptr + b * stride_vm_b + kv_h * stride_vm_h
    vscale_base = v_scale_ptr + b * stride_vm_b + kv_h * stride_vm_h
    tier_base = tier_out_ptr + b * stride_to_b + kv_h * stride_to_h

    for p in range(p_start, p_end):
        km_off = p * stride_km_p
        km0 = tl.load(kmin_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km1 = tl.load(kmin_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km2 = tl.load(kmin_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km3 = tl.load(kmin_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        ks0 = tl.load(kscale_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks1 = tl.load(kscale_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks2 = tl.load(kscale_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks3 = tl.load(kscale_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        bias_p = (
            tl.sum(q0 * (km0[None, :] + 1.5 * ks0[None, :]), axis=1)
            + tl.sum(q1 * (km1[None, :] + 1.5 * ks1[None, :]), axis=1)
            + tl.sum(q2 * (km2[None, :] + 1.5 * ks2[None, :]), axis=1)
            + tl.sum(q3 * (km3[None, :] + 1.5 * ks3[None, :]), axis=1)
        )
        p_byte_off = p * stride_kb_p + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4
        kb = tl.load(kbase_base + p_byte_off, mask=mask_d4[None, :], other=0)

        s_p = tl.broadcast_to(bias_p[:, None], (BLOCK_G, PAGE_SIZE))
        s_p = tl.dot((q0 * (4.0 * ks0[None, :])).to(tl.float16), tl.trans((kb & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q1 * (4.0 * ks1[None, :])).to(tl.float16), tl.trans(((kb >> 2) & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q2 * (4.0 * ks2[None, :])).to(tl.float16), tl.trans(((kb >> 4) & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q3 * (4.0 * ks3[None, :])).to(tl.float16), tl.trans(((kb >> 6) & 3).to(tl.float16)), acc=s_p)

        m_base_p = tl.max(s_p, axis=1)
        delta_page = tl.max(tl.where(mask_g, m_base_p - m_anchor, -1e20))

        need_page = (p == 0) or (delta_page > -tau_skip)
        is_hot = (p == 0) or (delta_page > -tau_hot)

        if RECORD_TIERS:
            t_code = 2 if is_hot else (1 if need_page else 0)
            tl.store(tier_base + p * stride_to_p, t_code)

        if need_page:
            if is_hot:
                # 热点页：直接复用寄存器 ks0..ks3 增量读取 K_res 修正至 4-bit
                qks0 = q0 * ks0[None, :]
                qks1 = q1 * ks1[None, :]
                qks2 = q2 * ks2[None, :]
                qks3 = q3 * ks3[None, :]
                res_bias = -1.5 * (
                    tl.sum(qks0, axis=1)
                    + tl.sum(qks1, axis=1)
                    + tl.sum(qks2, axis=1)
                    + tl.sum(qks3, axis=1)
                )
                kr = tl.load(kres_base + p_byte_off, mask=mask_d4[None, :], other=0)
                s_p += res_bias[:, None]
                s_p = tl.dot(qks0.to(tl.float16), tl.trans((kr & 3).to(tl.float16)), acc=s_p)
                s_p = tl.dot(qks1.to(tl.float16), tl.trans(((kr >> 2) & 3).to(tl.float16)), acc=s_p)
                s_p = tl.dot(qks2.to(tl.float16), tl.trans(((kr >> 4) & 3).to(tl.float16)), acc=s_p)
                s_p = tl.dot(qks3.to(tl.float16), tl.trans(((kr >> 6) & 3).to(tl.float16)), acc=s_p)

            m_ij_eff = tl.max(s_p, axis=1)
            m_new = tl.maximum(m_i, m_ij_eff)
            alpha = tl.where(l_i == 0.0, 0.0, tl.exp(m_i - m_new))
            p_val = tl.exp(s_p - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p_val, axis=1)
            m_i = m_new

            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc2 = acc2 * alpha[:, None]
            acc3 = acc3 * alpha[:, None]

            vm = tl.load(vmin_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
            vs = tl.load(vscale_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
            vb = tl.load(vbase_base + p_byte_off, mask=mask_d4[None, :], other=0)

            if is_hot:
                vr = tl.load(vres_base + p_byte_off, mask=mask_d4[None, :], other=0)
                scalar_bias = tl.sum(p_val * vm[None, :], axis=1)[:, None]
                p_sc = p_val * vs[None, :]
                row_scale = tl.maximum(tl.max(tl.abs(p_sc), axis=1), 1e-12)
                p_sc_n = (p_sc / row_scale[:, None]).to(tl.float16)

                acc0 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * (vb & 3).to(tl.float16) + (vr & 3).to(tl.float16))
                acc1 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 2) & 3).to(tl.float16) + ((vr >> 2) & 3).to(tl.float16))
                acc2 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 4) & 3).to(tl.float16) + ((vr >> 4) & 3).to(tl.float16))
                acc3 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 6) & 3).to(tl.float16) + ((vr >> 6) & 3).to(tl.float16))
            else:
                scalar_bias = tl.sum(p_val * (vm[None, :] + 1.5 * vs[None, :]), axis=1)[:, None]
                p_sc = 4.0 * p_val * vs[None, :]
                row_scale = tl.maximum(tl.max(tl.abs(p_sc), axis=1), 1e-12)
                p_sc_n = (p_sc / row_scale[:, None]).to(tl.float16)

                acc0 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, (vb & 3).to(tl.float16))
                acc1 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 2) & 3).to(tl.float16))
                acc2 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 4) & 3).to(tl.float16))
                acc3 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 6) & 3).to(tl.float16))

    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(pm_out, m_i, mask=mask_g)
    tl.store(pl_out, l_i, mask=mask_g)
    tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, acc0 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, acc1 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, acc2 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, acc3 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])


def triton_bitplane_gated_decode(
    query: torch.Tensor,
    cache: PackedBitPlaneKVCache,
    scaling: float,
    tau_hot: float = 2.0,
    tau_skip: float = 4.5,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """
    Single-Pass Anchor-Gated Bit-Plane 融合解码核（零中间表读写、零 Top-k、三 Launch 全闭环）。

    Args:
        query: (B, H_q, 1, d)
        cache: PackedBitPlaneKVCache
        scaling: 1 / sqrt(d)
        tau_hot: Hot (4-bit) 门控阈值（默认 2.0）
        tau_skip: Skip (免读 Value) 门控阈值（默认 4.5）
        return_stats: 为 True 时记录并返回实际触发的 hot/warm/skip 比例与物理读取比特
    """
    if query.shape[2] != 1:
        raise ValueError(f"解码 Triton 核要求 L_q == 1，实际收到 {query.shape[2]}")

    bsz, h_q, _, head_dim = query.shape
    h_kv = cache.num_kv_heads
    grp = h_q // h_kv
    num_pages = cache.num_pages
    page_size = cache.page_size
    tail_len = cache.tail_len
    has_tail = tail_len > 0

    if num_pages == 0:
        if not has_tail:
            raise ValueError("缓存中既无历史页也无尾部全精 KV")
        k_t = cache.k_tail[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        v_t = cache.v_tail[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        out = F.scaled_dot_product_attention(query.float(), k_t.float(), v_t.float(), scale=scaling).transpose(1, 2).contiguous().to(query.dtype)
        if return_stats:
            return out, {"hot_ratio": 0.0, "warm_ratio": 0.0, "skip_ratio": 0.0, "phys_read_bits": 16.0}
        return out

    d4 = head_dim // 4
    block_g = max(16, triton.next_power_of_2(grp))
    block_d4 = max(16, triton.next_power_of_2(d4))
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_t = max(16, triton.next_power_of_2(max(1, tail_len)))
    num_splits = _choose_num_splits(bsz, h_kv, num_pages)
    device = query.device

    q_c = query.squeeze(2).contiguous()
    tail_m = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    tail_l = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    tail_o = torch.empty((bsz, h_q, head_dim), dtype=torch.float32, device=device)
    m_anchor = torch.empty((bsz, h_q), dtype=torch.float32, device=device)

    use_kernel_tail = (not has_tail) or (tail_len <= 64)
    kt_ref = cache.k_tail if (has_tail and use_kernel_tail) else q_c[:, :h_kv, :1]
    vt_ref = cache.v_tail if (has_tail and use_kernel_tail) else q_c[:, :h_kv, :1]

    _anchor_and_tail_kernel[(bsz * h_kv,)](
        q_c,
        cache.k_base_packed,
        cache.k_min,
        cache.k_scale,
        kt_ref,
        vt_ref,
        tail_m,
        tail_l,
        tail_o,
        m_anchor,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        cache.k_base_packed.stride(0),
        cache.k_base_packed.stride(1),
        cache.k_base_packed.stride(3),
        cache.k_base_packed.stride(4),
        cache.k_min.stride(0),
        cache.k_min.stride(1),
        cache.k_min.stride(3),
        kt_ref.stride(0),
        kt_ref.stride(1),
        kt_ref.stride(2) if kt_ref.ndim == 4 else 1,
        kt_ref.stride(3) if kt_ref.ndim == 4 else 1,
        tail_m.stride(0),
        tail_m.stride(1),
        tail_o.stride(0),
        tail_o.stride(1),
        tail_o.stride(2),
        scaling,
        tail_len,
        H_kv=h_kv,
        G=grp,
        HEAD_DIM=head_dim,
        D4=d4,
        PAGE_SIZE=page_size,
        BLOCK_G=block_g,
        BLOCK_D=block_d,
        BLOCK_D4=block_d4,
        BLOCK_T=block_t,
        HAS_TAIL=has_tail,
        COMPUTE_TAIL=use_kernel_tail,
        num_warps=4,
        num_stages=1,
    )

    if has_tail and (not use_kernel_tail):
        q_g = query.squeeze(2).view(bsz, h_kv, grp, head_dim).float()
        scores_t = torch.matmul(q_g, cache.k_tail.float().transpose(-2, -1)) * scaling
        tm_g = scores_t.max(dim=-1).values
        exp_t = torch.exp(scores_t - tm_g.unsqueeze(-1))
        tl_g = exp_t.sum(dim=-1)
        to_g = torch.matmul(exp_t / torch.clamp(tl_g.unsqueeze(-1), min=1e-30), cache.v_tail.float())
        tail_m = tm_g.reshape(bsz, h_q)
        tail_l = tl_g.reshape(bsz, h_q)
        tail_o = to_g.reshape(bsz, h_q, head_dim)
        m_anchor = torch.maximum(m_anchor, tail_m)

    part_m = torch.empty((bsz, h_q, num_splits), dtype=torch.float32, device=device)
    part_l = torch.empty((bsz, h_q, num_splits), dtype=torch.float32, device=device)
    part_o = torch.empty((bsz, h_q, num_splits, head_dim), dtype=torch.float32, device=device)

    if return_stats:
        tier_out = torch.empty((bsz, h_kv, num_pages), dtype=torch.uint8, device=device)
    else:
        tier_out = torch.empty((1, 1, 1), dtype=torch.uint8, device=device)

    _bitplane_singlepass_gated_kernel[(bsz * h_kv, num_splits)](
        q_c,
        m_anchor,
        cache.k_base_packed,
        cache.k_res_packed,
        cache.k_min,
        cache.k_scale,
        cache.v_base_packed,
        cache.v_res_packed,
        cache.v_min,
        cache.v_scale,
        part_m,
        part_l,
        part_o,
        tier_out,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        m_anchor.stride(0),
        m_anchor.stride(1),
        cache.k_base_packed.stride(0),
        cache.k_base_packed.stride(1),
        cache.k_base_packed.stride(2),
        cache.k_base_packed.stride(3),
        cache.k_base_packed.stride(4),
        cache.k_min.stride(0),
        cache.k_min.stride(1),
        cache.k_min.stride(2),
        cache.k_min.stride(3),
        cache.v_min.stride(0),
        cache.v_min.stride(1),
        cache.v_min.stride(2),
        cache.v_min.stride(3),
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        tier_out.stride(0),
        tier_out.stride(1),
        tier_out.stride(2),
        scaling,
        float(tau_hot),
        float(tau_skip),
        num_pages,
        num_splits,
        H_kv=h_kv,
        G=grp,
        D4=d4,
        PAGE_SIZE=page_size,
        BLOCK_G=block_g,
        BLOCK_D4=block_d4,
        RECORD_TIERS=return_stats,
        num_warps=4,
    )

    out_fp32 = torch.empty((bsz, 1, h_q, head_dim), dtype=torch.float32, device=device)
    out_v = out_fp32.squeeze(1)
    _splitk_reduce_kernel[(bsz * h_q,)](
        part_m,
        part_l,
        part_o,
        tail_m,
        tail_l,
        tail_o,
        out_v,
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        tail_m.stride(0),
        tail_m.stride(1),
        tail_o.stride(0),
        tail_o.stride(1),
        tail_o.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        num_splits,
        H_q=h_q,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        HAS_TAIL=has_tail,
    )
    out = out_fp32.to(query.dtype)

    if return_stats:
        hot_r = float((tier_out == 2).float().mean().item())
        warm_r = float((tier_out == 1).float().mean().item())
        skip_r = float((tier_out == 0).float().mean().item())
        phys_bits = 3.0 * skip_r + 5.25 * warm_r + 9.25 * hot_r
        return out, {
            "hot_ratio": hot_r,
            "warm_ratio": warm_r,
            "skip_ratio": skip_r,
            "phys_read_bits": phys_bits,
            "num_splits": num_splits,
        }
    return out


_SM_COUNT_CACHE: dict[int, int] = {}


def _get_sm_count(device: torch.device) -> int:
    """读取并缓存当前 CUDA 设备的流式多处理器 (SM) 数量，热路径零开销。"""
    dev_idx = 0 if device.index is None else int(device.index)
    cached = _SM_COUNT_CACHE.get(dev_idx)
    if cached is not None:
        return cached
    cnt = int(torch.cuda.get_device_properties(device).multi_processor_count)
    _SM_COUNT_CACHE[dev_idx] = cnt
    return cnt


def _choose_dual_splits(
    batch_size: int,
    num_kv_heads: int,
    num_sel_pages: int,
    head_dim: int = 128,
    sm_count: int = 70,
) -> int:
    """
    为 Zero-Branch 双流核自适应分配 Wave-Aligned Split-KV 分片数。
    采用向下整除 SM 并行波容量（280 CTAs @ d=128 / 140 CTAs @ d=256），杜绝尾波溢出。
    """
    if num_sel_pages <= 0:
        return 0
    if num_sel_pages == 1:
        return 1
    ctas_per_sm = 2 if head_dim >= 256 else 4
    wave_cap = max(1, sm_count * ctas_per_sm)
    total_groups = max(1, batch_size * num_kv_heads)
    target_waves = 2 if num_sel_pages >= 128 else 1
    wave_aligned = max(1, (target_waves * wave_cap) // total_groups)
    max_by_pages = max(1, num_sel_pages // 8)
    if num_sel_pages < 32:
        max_by_pages = max(1, num_sel_pages // 2)
    return max(1, min(48, wave_aligned, max_by_pages))


@triton.jit
def _ellipsoid_score_kernel(
    q_ptr,
    k_min_ptr,
    k_scale_ptr,
    head_scores_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_km_b,
    stride_km_h,
    stride_km_p,
    stride_km_d,
    stride_hs_b,
    stride_hs_h,
    stride_hs_p,
    scaling,
    beta,
    num_pages,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    USE_ELLIPSOID: tl.constexpr,
):
    """
    零额外元数据椭球统计页打分核（Grid = (B * H_kv, cdiv(P, BLOCK_P))）：
      复用已有页尺度的 (k_min, k_scale) 构造中心 k_mid = k_min + 7.5*k_scale 与半径 7.5*k_scale，
      通过 FP16 Tensor Core 算出每头页上界期望 mu 及按行归一化防溢出的标准差修正。
    """
    b_kv = tl.program_id(0)
    p_blk = tl.program_id(1)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    offs_p = p_blk * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < num_pages

    q_tile = tl.load(
        q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh + offs_d[None, :] * stride_qd,
        mask=mask_g[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32) * scaling

    p_meta_off = b * stride_km_b + kv_h * stride_km_h + offs_p[:, None] * stride_km_p + offs_d[None, :] * stride_km_d
    km = tl.load(k_min_ptr + p_meta_off, mask=mask_p[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    ks = tl.load(k_scale_ptr + p_meta_off, mask=mask_p[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    kmid_fp16 = (km + 7.5 * ks).to(tl.float16)
    mu = tl.dot(q_tile.to(tl.float16), tl.trans(kmid_fp16))

    if USE_ELLIPSOID:
        # 行极大归一化使平方输入严格落入 [0, 1]，杜绝离群通道平方进 FP16 Tensor Core 上下溢
        q_max = tl.maximum(tl.max(tl.abs(q_tile), axis=1), 1e-6)
        q_n = q_tile / q_max[:, None]
        q_n2_fp16 = (q_n * q_n).to(tl.float16)

        r_max = tl.maximum(tl.max(tl.abs(ks), axis=1), 1e-6)
        ks_n = ks / r_max[:, None]
        ks_n2_fp16 = (ks_n * ks_n).to(tl.float16)

        var_n = tl.dot(q_n2_fp16, tl.trans(ks_n2_fp16))
        u = mu + beta * (7.5 * q_max[:, None] * r_max[None, :]) * tl.sqrt(tl.maximum(var_n, 0.0))
    else:
        u = mu

    tl.store(
        head_scores_ptr + b * stride_hs_b + (kv_h * G + offs_g[:, None]) * stride_hs_h + offs_p[None, :] * stride_hs_p,
        u,
        mask=mask_g[:, None] & mask_p[None, :],
    )


@triton.jit
def _dualstream_agg_and_tail_kernel(
    q_ptr,
    k_tail_ptr,
    v_tail_ptr,
    head_scores_ptr,
    tail_m_ptr,
    tail_l_ptr,
    tail_o_ptr,
    grp_scores_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kt_b,
    stride_kt_h,
    stride_kt_s,
    stride_kt_d,
    stride_hs_b,
    stride_hs_h,
    stride_hs_p,
    stride_tm_b,
    stride_tm_h,
    stride_to_b,
    stride_to_h,
    stride_to_d,
    stride_gs_b,
    stride_gs_h,
    stride_gs_p,
    scaling,
    num_pages,
    tail_len,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    COMPUTE_TAIL: tl.constexpr,
    PIN_PAGE0: tl.constexpr,
):
    """
    融合尾部全精计算与 GQA 组相对极值聚合单核（Grid = (B * H_kv,)）：
      1. 若 HAS_TAIL and COMPUTE_TAIL：使用 Tensor Core 计算尾部全精统计 (tail_m, tail_l, tail_o)；
      2. 对 head_scores (B, H_q, P) 减去逐头最大值并沿 GQA 组取 max，可选固定 Page 0 进 Hot 队列。
    """
    b_kv = tl.program_id(0)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    if HAS_TAIL and COMPUTE_TAIL:
        offs_t = tl.arange(0, BLOCK_T)
        mask_t = offs_t < tail_len
        q_tile = tl.load(
            q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh + offs_d[None, :] * stride_qd,
            mask=mask_g[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32) * scaling
        kt_tile = tl.load(
            k_tail_ptr + b * stride_kt_b + kv_h * stride_kt_h + offs_t[:, None] * stride_kt_s + offs_d[None, :] * stride_kt_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        s_t = tl.dot(q_tile.to(tl.float16), tl.trans(kt_tile.to(tl.float16)))
        s_t = tl.where(mask_t[None, :], s_t, -1e20)

        m_t = tl.max(s_t, axis=1)
        exp_t = tl.where(mask_t[None, :], tl.exp(s_t - m_t[:, None]), 0.0)
        l_t = tl.sum(exp_t, axis=1)
        w_t = exp_t / tl.maximum(l_t[:, None], 1e-30)
        vt_tile = tl.load(
            v_tail_ptr + b * stride_kt_b + kv_h * stride_kt_h + offs_t[:, None] * stride_kt_s + offs_d[None, :] * stride_kt_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        o_t = tl.dot(w_t.to(tl.float16), vt_tile.to(tl.float16))

        tl.store(tail_m_ptr + b * stride_tm_b + (kv_h * G + offs_g) * stride_tm_h, m_t, mask=mask_g)
        tl.store(tail_l_ptr + b * stride_tm_b + (kv_h * G + offs_g) * stride_tm_h, l_t, mask=mask_g)
        tl.store(
            tail_o_ptr + b * stride_to_b + (kv_h * G + offs_g[:, None]) * stride_to_h + offs_d[None, :] * stride_to_d,
            o_t,
            mask=mask_g[:, None] & mask_d[None, :],
        )

    hs_base = head_scores_ptr + b * stride_hs_b + (kv_h * G + offs_g[:, None]) * stride_hs_h
    gs_base = grp_scores_ptr + b * stride_gs_b + kv_h * stride_gs_h

    # Pass 1：逐头求全页最大得分 h_max
    h_max = tl.full((BLOCK_G,), -1e20, tl.float32)
    offs_p = tl.arange(0, BLOCK_P)
    for start_p in range(0, num_pages, BLOCK_P):
        p_idx = start_p + offs_p
        mask_p = p_idx < num_pages
        hs = tl.load(hs_base + p_idx[None, :] * stride_hs_p, mask=mask_g[:, None] & mask_p[None, :], other=-1e20)
        h_max = tl.maximum(h_max, tl.max(hs, axis=1))

    # Pass 2：减去每头最大值后按 GQA 组取 max，保证组内各头同量纲竞争且共享页索引
    for start_p in range(0, num_pages, BLOCK_P):
        p_idx = start_p + offs_p
        mask_p = p_idx < num_pages
        hs = tl.load(hs_base + p_idx[None, :] * stride_hs_p, mask=mask_g[:, None] & mask_p[None, :], other=-1e20)
        u_rel = tl.where(mask_g[:, None] & mask_p[None, :], hs - h_max[:, None], -1e20)
        grp_p = tl.max(u_rel, axis=0)
        if PIN_PAGE0:
            grp_p = tl.where(p_idx == 0, 1e9, grp_p)
        tl.store(gs_base + p_idx * stride_gs_p, grp_p, mask=mask_p)


@triton.jit
def _indexed_int4_splitkv_kernel(
    q_ptr,
    page_idx_ptr,
    k_base_ptr,
    k_res_ptr,
    k_min_ptr,
    k_scale_ptr,
    v_base_ptr,
    v_res_ptr,
    v_min_ptr,
    v_scale_ptr,
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_pi_b,
    stride_pi_h,
    stride_pi_k,
    stride_kb_b,
    stride_kb_h,
    stride_kb_p,
    stride_kb_s,
    stride_kb_d4,
    stride_km_b,
    stride_km_h,
    stride_km_p,
    stride_km_d,
    stride_vm_b,
    stride_vm_h,
    stride_vm_p,
    stride_vm_s,
    stride_pm_b,
    stride_pm_h,
    stride_pm_k,
    stride_po_b,
    stride_po_h,
    stride_po_k,
    stride_po_d,
    scaling,
    num_sel_pages,
    num_splits,
    split_offset,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    D4: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D4: tl.constexpr,
):
    """Zero-Branch Hot 流核：无循环内分支，仅按 page_idx 紧凑读取并精确计算 4-bit 热点页。"""
    b_kv = tl.program_id(0)
    split_id = tl.program_id(1)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    pages_per_split = tl.cdiv(num_sel_pages, num_splits)
    i_start = split_id * pages_per_split
    i_end = tl.minimum(num_sel_pages, i_start + pages_per_split)
    out_slot = split_offset + split_id

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d4 = tl.arange(0, BLOCK_D4)
    mask_d4 = offs_d4 < D4
    offs_s = tl.arange(0, PAGE_SIZE)

    pm_out = part_m_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + out_slot * stride_pm_k
    pl_out = part_l_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + out_slot * stride_pm_k
    po_out = part_o_ptr + b * stride_po_b + (kv_h * G + offs_g[:, None]) * stride_po_h + out_slot * stride_po_k

    if i_start >= i_end:
        tl.store(pm_out, tl.full((BLOCK_G,), -1e20, tl.float32), mask=mask_g)
        tl.store(pl_out, tl.zeros((BLOCK_G,), tl.float32), mask=mask_g)
        z_o = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
        tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        return

    q_row_ptr = q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh
    q0 = tl.load(q_row_ptr + (0 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_row_ptr + (1 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_row_ptr + (2 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_row_ptr + (3 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling

    m_i = tl.full((BLOCK_G,), -1e20, tl.float32)
    l_i = tl.zeros((BLOCK_G,), tl.float32)
    acc0 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc1 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc2 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc3 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)

    pi_base = page_idx_ptr + b * stride_pi_b + kv_h * stride_pi_h
    kbase_base = k_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    kres_base = k_res_ptr + b * stride_kb_b + kv_h * stride_kb_h
    kmin_base = k_min_ptr + b * stride_km_b + kv_h * stride_km_h
    kscale_base = k_scale_ptr + b * stride_km_b + kv_h * stride_km_h
    vbase_base = v_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vres_base = v_res_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vmin_base = v_min_ptr + b * stride_vm_b + kv_h * stride_vm_h
    vscale_base = v_scale_ptr + b * stride_vm_b + kv_h * stride_vm_h

    for i in range(i_start, i_end):
        # 强制 int64 页索引乘法，防止大上下文长度 (>32K) 下指针字节偏移溢出 int32
        p = tl.load(pi_base + i * stride_pi_k).to(tl.int64)
        km_off = p * stride_km_p
        km0 = tl.load(kmin_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km1 = tl.load(kmin_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km2 = tl.load(kmin_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km3 = tl.load(kmin_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        ks0 = tl.load(kscale_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks1 = tl.load(kscale_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks2 = tl.load(kscale_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks3 = tl.load(kscale_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        bias_k = (
            tl.sum(q0 * km0[None, :], axis=1)
            + tl.sum(q1 * km1[None, :], axis=1)
            + tl.sum(q2 * km2[None, :], axis=1)
            + tl.sum(q3 * km3[None, :], axis=1)
        )

        p_byte_off = p * stride_kb_p + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4
        kb = tl.load(kbase_base + p_byte_off, mask=mask_d4[None, :], other=0)
        kr = tl.load(kres_base + p_byte_off, mask=mask_d4[None, :], other=0)

        s_p = tl.broadcast_to(bias_k[:, None], (BLOCK_G, PAGE_SIZE))
        s_p = tl.dot((q0 * ks0[None, :]).to(tl.float16), tl.trans(4.0 * (kb & 3).to(tl.float16) + (kr & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q1 * ks1[None, :]).to(tl.float16), tl.trans(4.0 * ((kb >> 2) & 3).to(tl.float16) + ((kr >> 2) & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q2 * ks2[None, :]).to(tl.float16), tl.trans(4.0 * ((kb >> 4) & 3).to(tl.float16) + ((kr >> 4) & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q3 * ks3[None, :]).to(tl.float16), tl.trans(4.0 * ((kb >> 6) & 3).to(tl.float16) + ((kr >> 6) & 3).to(tl.float16)), acc=s_p)

        m_ij = tl.max(s_p, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(l_i == 0.0, 0.0, tl.exp(m_i - m_new))
        p_val = tl.exp(s_p - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p_val, axis=1)
        m_i = m_new

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        acc3 = acc3 * alpha[:, None]

        vm = tl.load(vmin_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
        vs = tl.load(vscale_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
        vb = tl.load(vbase_base + p_byte_off, mask=mask_d4[None, :], other=0)
        vr = tl.load(vres_base + p_byte_off, mask=mask_d4[None, :], other=0)

        scalar_bias = tl.sum(p_val * vm[None, :], axis=1)[:, None]
        p_sc = p_val * vs[None, :]
        row_scale = tl.maximum(tl.max(tl.abs(p_sc), axis=1), 1e-12)
        p_sc_n = (p_sc / row_scale[:, None]).to(tl.float16)

        acc0 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * (vb & 3).to(tl.float16) + (vr & 3).to(tl.float16))
        acc1 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 2) & 3).to(tl.float16) + ((vr >> 2) & 3).to(tl.float16))
        acc2 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 4) & 3).to(tl.float16) + ((vr >> 4) & 3).to(tl.float16))
        acc3 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, 4.0 * ((vb >> 6) & 3).to(tl.float16) + ((vr >> 6) & 3).to(tl.float16))

    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(pm_out, m_i, mask=mask_g)
    tl.store(pl_out, l_i, mask=mask_g)
    tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, acc0 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, acc1 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, acc2 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, acc3 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])


@triton.jit
def _indexed_base2_splitkv_kernel(
    q_ptr,
    page_idx_ptr,
    k_base_ptr,
    k_min_ptr,
    k_scale_ptr,
    v_base_ptr,
    v_min_ptr,
    v_scale_ptr,
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_pi_b,
    stride_pi_h,
    stride_pi_k,
    stride_kb_b,
    stride_kb_h,
    stride_kb_p,
    stride_kb_s,
    stride_kb_d4,
    stride_km_b,
    stride_km_h,
    stride_km_p,
    stride_km_d,
    stride_vm_b,
    stride_vm_h,
    stride_vm_p,
    stride_vm_s,
    stride_pm_b,
    stride_pm_h,
    stride_pm_k,
    stride_po_b,
    stride_po_h,
    stride_po_k,
    stride_po_d,
    scaling,
    num_sel_pages,
    num_splits,
    split_offset,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    D4: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D4: tl.constexpr,
):
    """Zero-Branch Warm 流核：仅读 2-bit Base 位面与尺度的超轻仿射预折叠解码核，彻底释放 cp.async 异步流水线。"""
    b_kv = tl.program_id(0)
    split_id = tl.program_id(1)
    b = b_kv // H_kv
    kv_h = b_kv % H_kv

    pages_per_split = tl.cdiv(num_sel_pages, num_splits)
    i_start = split_id * pages_per_split
    i_end = tl.minimum(num_sel_pages, i_start + pages_per_split)
    out_slot = split_offset + split_id

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d4 = tl.arange(0, BLOCK_D4)
    mask_d4 = offs_d4 < D4
    offs_s = tl.arange(0, PAGE_SIZE)

    pm_out = part_m_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + out_slot * stride_pm_k
    pl_out = part_l_ptr + b * stride_pm_b + (kv_h * G + offs_g) * stride_pm_h + out_slot * stride_pm_k
    po_out = part_o_ptr + b * stride_po_b + (kv_h * G + offs_g[:, None]) * stride_po_h + out_slot * stride_po_k

    if i_start >= i_end:
        tl.store(pm_out, tl.full((BLOCK_G,), -1e20, tl.float32), mask=mask_g)
        tl.store(pl_out, tl.zeros((BLOCK_G,), tl.float32), mask=mask_g)
        z_o = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
        tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, z_o, mask=mask_g[:, None] & mask_d4[None, :])
        return

    q_row_ptr = q_ptr + b * stride_qb + (kv_h * G + offs_g[:, None]) * stride_qh
    q0 = tl.load(q_row_ptr + (0 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q1 = tl.load(q_row_ptr + (1 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q2 = tl.load(q_row_ptr + (2 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling
    q3 = tl.load(q_row_ptr + (3 * D4 + offs_d4[None, :]) * stride_qd, mask=mask_g[:, None] & mask_d4[None, :], other=0.0).to(tl.float32) * scaling

    m_i = tl.full((BLOCK_G,), -1e20, tl.float32)
    l_i = tl.zeros((BLOCK_G,), tl.float32)
    acc0 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc1 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc2 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)
    acc3 = tl.zeros((BLOCK_G, BLOCK_D4), tl.float32)

    pi_base = page_idx_ptr + b * stride_pi_b + kv_h * stride_pi_h
    kbase_base = k_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    kmin_base = k_min_ptr + b * stride_km_b + kv_h * stride_km_h
    kscale_base = k_scale_ptr + b * stride_km_b + kv_h * stride_km_h
    vbase_base = v_base_ptr + b * stride_kb_b + kv_h * stride_kb_h
    vmin_base = v_min_ptr + b * stride_vm_b + kv_h * stride_vm_h
    vscale_base = v_scale_ptr + b * stride_vm_b + kv_h * stride_vm_h

    for i in range(i_start, i_end):
        p = tl.load(pi_base + i * stride_pi_k).to(tl.int64)
        km_off = p * stride_km_p
        km0 = tl.load(kmin_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km1 = tl.load(kmin_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km2 = tl.load(kmin_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        km3 = tl.load(kmin_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        ks0 = tl.load(kscale_base + km_off + (0 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks1 = tl.load(kscale_base + km_off + (1 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks2 = tl.load(kscale_base + km_off + (2 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)
        ks3 = tl.load(kscale_base + km_off + (3 * D4 + offs_d4) * stride_km_d, mask=mask_d4, other=0.0).to(tl.float32)

        bias_k = (
            tl.sum(q0 * (km0[None, :] + 1.5 * ks0[None, :]), axis=1)
            + tl.sum(q1 * (km1[None, :] + 1.5 * ks1[None, :]), axis=1)
            + tl.sum(q2 * (km2[None, :] + 1.5 * ks2[None, :]), axis=1)
            + tl.sum(q3 * (km3[None, :] + 1.5 * ks3[None, :]), axis=1)
        )

        p_byte_off = p * stride_kb_p + offs_s[:, None] * stride_kb_s + offs_d4[None, :] * stride_kb_d4
        kb = tl.load(kbase_base + p_byte_off, mask=mask_d4[None, :], other=0)

        s_p = tl.broadcast_to(bias_k[:, None], (BLOCK_G, PAGE_SIZE))
        s_p = tl.dot((q0 * (4.0 * ks0[None, :])).to(tl.float16), tl.trans((kb & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q1 * (4.0 * ks1[None, :])).to(tl.float16), tl.trans(((kb >> 2) & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q2 * (4.0 * ks2[None, :])).to(tl.float16), tl.trans(((kb >> 4) & 3).to(tl.float16)), acc=s_p)
        s_p = tl.dot((q3 * (4.0 * ks3[None, :])).to(tl.float16), tl.trans(((kb >> 6) & 3).to(tl.float16)), acc=s_p)

        m_ij = tl.max(s_p, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(l_i == 0.0, 0.0, tl.exp(m_i - m_new))
        p_val = tl.exp(s_p - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p_val, axis=1)
        m_i = m_new

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        acc3 = acc3 * alpha[:, None]

        vm = tl.load(vmin_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
        vs = tl.load(vscale_base + p * stride_vm_p + offs_s * stride_vm_s).to(tl.float32)
        vb = tl.load(vbase_base + p_byte_off, mask=mask_d4[None, :], other=0)

        scalar_bias = tl.sum(p_val * (vm[None, :] + 1.5 * vs[None, :]), axis=1)[:, None]
        p_sc = 4.0 * p_val * vs[None, :]
        row_scale = tl.maximum(tl.max(tl.abs(p_sc), axis=1), 1e-12)
        p_sc_n = (p_sc / row_scale[:, None]).to(tl.float16)

        acc0 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, (vb & 3).to(tl.float16))
        acc1 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 2) & 3).to(tl.float16))
        acc2 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 4) & 3).to(tl.float16))
        acc3 += scalar_bias + row_scale[:, None] * tl.dot(p_sc_n, ((vb >> 6) & 3).to(tl.float16))

    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(pm_out, m_i, mask=mask_g)
    tl.store(pl_out, l_i, mask=mask_g)
    tl.store(po_out + (0 * D4 + offs_d4[None, :]) * stride_po_d, acc0 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (1 * D4 + offs_d4[None, :]) * stride_po_d, acc1 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (2 * D4 + offs_d4[None, :]) * stride_po_d, acc2 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])
    tl.store(po_out + (3 * D4 + offs_d4[None, :]) * stride_po_d, acc3 * inv_l[:, None], mask=mask_g[:, None] & mask_d4[None, :])


def triton_bitplane_dualstream_decode(
    query: torch.Tensor,
    cache: PackedBitPlaneKVCache,
    scaling: float,
    hot_ratio: float = 0.15,
    warm_ratio: float = 0.35,
    ellipsoid_beta: float = 1.5,
    pin_page0: bool = True,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """
    Zero-Branch Ellipsoid-Routed Dual-Stream Bit-Plane 融合解码核（终极高速主方法）。

    核心设计：
      1. 零额外存储的椭球统计页打分核仅扫 (k_min, k_scale) 元数据选页；
      2. 互斥索引切片驱动两个完全无分支（cp.async 满流水线）的 Split-KV 核：
         Hot 流读取 4-bit 热点页，Warm 流超轻读取 2-bit Base 长尾页；
      3. 统一通过预分配分片缓存交由单次 Reduce 核合并 Hot + Warm + 全精尾部。
    """
    if query.shape[2] != 1:
        raise ValueError(f"解码 Triton 核要求 L_q == 1，实际收到 {query.shape[2]}")

    bsz, h_q, _, head_dim = query.shape
    h_kv = cache.num_kv_heads
    grp = h_q // h_kv
    num_pages = cache.num_pages
    page_size = cache.page_size
    tail_len = cache.tail_len
    has_tail = tail_len > 0

    if num_pages == 0:
        if not has_tail:
            raise ValueError("缓存中既无历史页也无尾部全精 KV")
        k_t = cache.k_tail[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        v_t = cache.v_tail[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        out = F.scaled_dot_product_attention(query.float(), k_t.float(), v_t.float(), scale=scaling).transpose(1, 2).contiguous().to(query.dtype)
        if return_stats:
            return out, {
                "hot_ratio": 0.0,
                "warm_ratio": 0.0,
                "skip_ratio": 0.0,
                "phys_read_bits": 16.0,
                "num_splits_hot": 0,
                "num_splits_warm": 0,
            }
        return out

    n_hot = 0 if hot_ratio <= 0.0 else max(1, min(num_pages, int(math.ceil(num_pages * hot_ratio))))
    n_warm = 0 if warm_ratio <= 0.0 else max(0, min(num_pages - n_hot, int(math.ceil(num_pages * warm_ratio))))
    n_keep = n_hot + n_warm
    if n_keep <= 0:
        raise ValueError("hot_ratio 与 warm_ratio 不能同时为 0")

    # 当 Hot 覆盖全部页时，无需椭球打分直接走满页 INT4 高速核
    if n_hot == num_pages:
        out = triton_int4_paged_decode(query, cache, scaling)
        if return_stats:
            return out, {
                "hot_ratio": 1.0,
                "warm_ratio": 0.0,
                "skip_ratio": 0.0,
                "phys_read_bits": 9.25,
                "num_splits_hot": _choose_num_splits(bsz, h_kv, num_pages),
                "num_splits_warm": 0,
            }
        return out

    d4 = head_dim // 4
    block_g = max(16, triton.next_power_of_2(grp))
    block_d4 = max(16, triton.next_power_of_2(d4))
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_t = max(16, triton.next_power_of_2(max(1, tail_len)))
    device = query.device

    q_c = query.squeeze(2).contiguous()
    head_scores = torch.empty((bsz, h_q, num_pages), dtype=torch.float32, device=device)
    grp_scores = torch.empty((bsz, h_kv, num_pages), dtype=torch.float32, device=device)

    block_p_score = 32
    score_stages = 2 if head_dim >= 256 else 3
    use_ellip = float(ellipsoid_beta) > 0.0
    _ellipsoid_score_kernel[(bsz * h_kv, triton.cdiv(num_pages, block_p_score))](
        q_c,
        cache.k_min,
        cache.k_scale,
        head_scores,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        cache.k_min.stride(0),
        cache.k_min.stride(1),
        cache.k_min.stride(2),
        cache.k_min.stride(3),
        head_scores.stride(0),
        head_scores.stride(1),
        head_scores.stride(2),
        scaling,
        float(ellipsoid_beta),
        num_pages,
        H_kv=h_kv,
        G=grp,
        HEAD_DIM=head_dim,
        BLOCK_G=block_g,
        BLOCK_D=block_d,
        BLOCK_P=block_p_score,
        USE_ELLIPSOID=use_ellip,
        num_warps=4,
        num_stages=score_stages,
    )

    tail_m = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    tail_l = torch.empty((bsz, h_q), dtype=torch.float32, device=device)
    tail_o = torch.empty((bsz, h_q, head_dim), dtype=torch.float32, device=device)

    use_kernel_tail = (not has_tail) or (tail_len <= 64)
    kt_ref = cache.k_tail if (has_tail and use_kernel_tail) else q_c[:, :h_kv, :1]
    vt_ref = cache.v_tail if (has_tail and use_kernel_tail) else q_c[:, :h_kv, :1]

    _dualstream_agg_and_tail_kernel[(bsz * h_kv,)](
        q_c,
        kt_ref,
        vt_ref,
        head_scores,
        tail_m,
        tail_l,
        tail_o,
        grp_scores,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        kt_ref.stride(0),
        kt_ref.stride(1),
        kt_ref.stride(2) if kt_ref.ndim == 4 else 1,
        kt_ref.stride(3) if kt_ref.ndim == 4 else 1,
        head_scores.stride(0),
        head_scores.stride(1),
        head_scores.stride(2),
        tail_m.stride(0),
        tail_m.stride(1),
        tail_o.stride(0),
        tail_o.stride(1),
        tail_o.stride(2),
        grp_scores.stride(0),
        grp_scores.stride(1),
        grp_scores.stride(2),
        scaling,
        num_pages,
        tail_len,
        H_kv=h_kv,
        G=grp,
        HEAD_DIM=head_dim,
        BLOCK_G=block_g,
        BLOCK_D=block_d,
        BLOCK_T=block_t,
        BLOCK_P=256,
        HAS_TAIL=has_tail,
        COMPUTE_TAIL=use_kernel_tail,
        PIN_PAGE0=bool(pin_page0),
        num_warps=4,
    )

    if has_tail and (not use_kernel_tail):
        q_g = query.squeeze(2).view(bsz, h_kv, grp, head_dim).float()
        scores_t = torch.matmul(q_g, cache.k_tail.float().transpose(-2, -1)) * scaling
        tm_g = scores_t.max(dim=-1).values
        exp_t = torch.exp(scores_t - tm_g.unsqueeze(-1))
        tl_g = exp_t.sum(dim=-1)
        to_g = torch.matmul(exp_t / torch.clamp(tl_g.unsqueeze(-1), min=1e-30), cache.v_tail.float())
        tail_m = tm_g.reshape(bsz, h_q)
        tail_l = tl_g.reshape(bsz, h_q)
        tail_o = to_g.reshape(bsz, h_q, head_dim)

    # 紧凑互斥路由表（零 .contiguous() 额外拷贝，直接传 stride）
    top_idx = torch.topk(grp_scores, k=n_keep, dim=-1, sorted=True).indices.to(torch.int32)
    hot_idx = top_idx[:, :, :n_hot]
    warm_idx = top_idx[:, :, n_hot:n_keep]

    sm_cnt = _get_sm_count(device)
    num_splits_hot = _choose_dual_splits(bsz, h_kv, n_hot, head_dim=head_dim, sm_count=sm_cnt)
    num_splits_warm = _choose_dual_splits(bsz, h_kv, n_warm, head_dim=head_dim, sm_count=sm_cnt)
    total_splits = num_splits_hot + num_splits_warm

    part_m = torch.empty((bsz, h_q, total_splits), dtype=torch.float32, device=device)
    part_l = torch.empty((bsz, h_q, total_splits), dtype=torch.float32, device=device)
    part_o = torch.empty((bsz, h_q, total_splits, head_dim), dtype=torch.float32, device=device)

    if n_hot > 0:
        _indexed_int4_splitkv_kernel[(bsz * h_kv, num_splits_hot)](
            q_c,
            hot_idx,
            cache.k_base_packed,
            cache.k_res_packed,
            cache.k_min,
            cache.k_scale,
            cache.v_base_packed,
            cache.v_res_packed,
            cache.v_min,
            cache.v_scale,
            part_m,
            part_l,
            part_o,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            hot_idx.stride(0),
            hot_idx.stride(1),
            hot_idx.stride(2),
            cache.k_base_packed.stride(0),
            cache.k_base_packed.stride(1),
            cache.k_base_packed.stride(2),
            cache.k_base_packed.stride(3),
            cache.k_base_packed.stride(4),
            cache.k_min.stride(0),
            cache.k_min.stride(1),
            cache.k_min.stride(2),
            cache.k_min.stride(3),
            cache.v_min.stride(0),
            cache.v_min.stride(1),
            cache.v_min.stride(2),
            cache.v_min.stride(3),
            part_m.stride(0),
            part_m.stride(1),
            part_m.stride(2),
            part_o.stride(0),
            part_o.stride(1),
            part_o.stride(2),
            part_o.stride(3),
            scaling,
            n_hot,
            num_splits_hot,
            0,
            H_kv=h_kv,
            G=grp,
            D4=d4,
            PAGE_SIZE=page_size,
            BLOCK_G=block_g,
            BLOCK_D4=block_d4,
            num_warps=4,
        )

    if n_warm > 0:
        _indexed_base2_splitkv_kernel[(bsz * h_kv, num_splits_warm)](
            q_c,
            warm_idx,
            cache.k_base_packed,
            cache.k_min,
            cache.k_scale,
            cache.v_base_packed,
            cache.v_min,
            cache.v_scale,
            part_m,
            part_l,
            part_o,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            warm_idx.stride(0),
            warm_idx.stride(1),
            warm_idx.stride(2),
            cache.k_base_packed.stride(0),
            cache.k_base_packed.stride(1),
            cache.k_base_packed.stride(2),
            cache.k_base_packed.stride(3),
            cache.k_base_packed.stride(4),
            cache.k_min.stride(0),
            cache.k_min.stride(1),
            cache.k_min.stride(2),
            cache.k_min.stride(3),
            cache.v_min.stride(0),
            cache.v_min.stride(1),
            cache.v_min.stride(2),
            cache.v_min.stride(3),
            part_m.stride(0),
            part_m.stride(1),
            part_m.stride(2),
            part_o.stride(0),
            part_o.stride(1),
            part_o.stride(2),
            part_o.stride(3),
            scaling,
            n_warm,
            num_splits_warm,
            num_splits_hot,
            H_kv=h_kv,
            G=grp,
            D4=d4,
            PAGE_SIZE=page_size,
            BLOCK_G=block_g,
            BLOCK_D4=block_d4,
            num_warps=4,
        )

    out_fp32 = torch.empty((bsz, 1, h_q, head_dim), dtype=torch.float32, device=device)
    out_v = out_fp32.squeeze(1)
    _splitk_reduce_kernel[(bsz * h_q,)](
        part_m,
        part_l,
        part_o,
        tail_m,
        tail_l,
        tail_o,
        out_v,
        part_m.stride(0),
        part_m.stride(1),
        part_m.stride(2),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_o.stride(3),
        tail_m.stride(0),
        tail_m.stride(1),
        tail_o.stride(0),
        tail_o.stride(1),
        tail_o.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        total_splits,
        H_q=h_q,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        HAS_TAIL=has_tail,
    )
    out = out_fp32.to(query.dtype)

    if return_stats:
        hot_r = n_hot / num_pages
        warm_r = n_warm / num_pages
        skip_r = max(0.0, 1.0 - hot_r - warm_r)
        phys_bits = 1.0 + 9.25 * hot_r + 5.25 * warm_r
        return out, {
            "hot_ratio": hot_r,
            "warm_ratio": warm_r,
            "skip_ratio": skip_r,
            "phys_read_bits": phys_bits,
            "num_splits_hot": num_splits_hot,
            "num_splits_warm": num_splits_warm,
        }
    return out
