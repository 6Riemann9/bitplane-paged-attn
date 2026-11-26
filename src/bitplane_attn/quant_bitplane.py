"""
嵌套双位面 KV-Cache 量化与两阶段查询感知页级解码实现。

核心设计原则：
  1. 100% 兼容 PagedAttention 与 Prefix Caching：所有压缩、位面切分与索引均严格限于
     固定页大小 (page_size=32) 的纯前缀整页区间内，不依赖跨步未来状态、不驱逐任何 Token；
  2. Page-Aligned KIVI 分组粒度：Key 在页内按通道 (per-channel within each page) 量化以
     抑制通道离群与 RoPE 相位幅度波动；Value 按 Token (per-token) 量化；
  3. 硬件原生 4-bit 嵌套双位面：
       q4 = clamp(round((x - x_min) / delta_4), 0, 15)
       q_hi = q4 >> 2   (2-bit Base 位面)
       q_lo = q4 & 3    (2-bit Residual 位面)
     Base 无偏重构为 x_min + (4 * q_hi + 1.5) * delta_4；
     加载 Residual 后精修重构为 x_base + (q_lo - 1.5) * delta_4 === x_min + q4 * delta_4；
  4. 严格防因果泄密：多步连续评估 (L_q > 1) 时，每步 t 仅依据自身的 q_t 独立选页，
     且只能看到 k <= past_len + t。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def compute_logical_read_bits_per_kv_elem(
    mode: str,
    hot_ratio: float = 1.0,
    warm_ratio: float = 0.0,
    skip_ratio: float = 0.0,
    head_dim: int = 128,
    page_size: int = 32,
) -> float:
    """
    按每 KV 标量元素（bits / KV-elem，即单次解码理论逻辑读取总比特 / (2 * L * d)）统一核算读带宽。

    注：此处全部精确包含页尺度 float16 元数据：
      - k_meta = (2 * d * 16) / (2 * S * d) = 16.0 / page_size  （S=32 时为 0.50 bits/KV-elem）
      - v_meta = (2 * S * 16) / (2 * S * d) = 16.0 / head_dim   （d=128 时 0.125；d=256 时 0.0625）
      - 该指标刻画的是理论逻辑访存流大小，不等价于 GPU 物理 DRAM 硬件事务计数器实测字节数。
    """
    k_meta = 16.0 / float(max(1, page_size))
    v_meta = 16.0 / float(max(1, head_dim))
    if mode in ("fp16", "bf16"):
        return 16.0
    if mode == "int4":
        return 4.0 + k_meta + v_meta
    if mode in ("base2", "int2"):
        return 2.0 + k_meta + v_meta
    if mode == "bitplane":
        # Stage 1 全扫 K_base + k_meta (1.0 + k_meta)，Stage 2 对 Hot 读 3.0+v_meta，其余读 1.0+v_meta
        return 2.0 + k_meta + v_meta + 2.0 * float(hot_ratio)
    if mode == "bitplane_droptail":
        return 1.0 + k_meta + float(hot_ratio) * (3.0 + v_meta)
    if mode == "bitplane_k4_v2":
        return 2.0 + k_meta + v_meta + float(hot_ratio)
    if mode == "bitplane_gated":
        return (
            float(skip_ratio) * (1.0 + k_meta)
            + float(warm_ratio) * (2.0 + k_meta + v_meta)
            + float(hot_ratio) * (4.0 + k_meta + v_meta)
        )
    if mode == "bitplane_dualstream":
        # 路由核全扫 (k_min, k_scale) 占 k_meta；Hot 流读取完整 4-bit 页，Warm 流仅读取 2-bit Base 页
        return (
            k_meta
            + float(hot_ratio) * (4.0 + k_meta + v_meta)
            + float(warm_ratio) * (2.0 + k_meta + v_meta)
        )
    if mode == "quest":
        return k_meta + 16.0 * float(hot_ratio)
    if mode == "key_only_int2":
        return 9.0 + k_meta
    if mode == "val_only_int2":
        return 9.0 + v_meta
    raise ValueError(f"未知的逻辑读取位宽计算模式: {mode}")


@dataclass
class QuantizedBitPlanePages:
    """保存历史整页 Key/Value 的嵌套双位面及均匀 2-bit 表示（以浮点重构张量与整数位面共同校验）。"""

    page_size: int
    hist_len: int
    num_pages: int
    # 重构后的各精度表示，形状统一为 (B, H_kv, hist_len, d)
    k_base: torch.Tensor
    k_int4: torch.Tensor
    k_int2: torch.Tensor
    v_base: torch.Tensor
    v_int4: torch.Tensor
    v_int2: torch.Tensor
    # Quest 论文的页内原始通道 min/max 包围盒，形状 (B, H_kv, P, d)
    k_page_min: torch.Tensor
    k_page_max: torch.Tensor


def _quantize_nested_tensors(
    x: torch.Tensor,
    reduce_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    沿指定维度对张量执行嵌套 4-bit 双位面切分，并返回：
      (q_hi, q_lo, x_min, delta_4, x_base_hat, x_int4_hat)

    其中 x_base_hat 与 x_int4_hat 使用严格位面算术重建，用于验证无偏性与精确一致性。
    """
    x_fp32 = x.float()
    x_min = x_fp32.min(dim=reduce_dim, keepdim=True).values
    x_max = x_fp32.max(dim=reduce_dim, keepdim=True).values
    delta_4 = torch.clamp(x_max - x_min, min=1e-8) / 15.0

    q4 = torch.clamp(torch.round((x_fp32 - x_min) / delta_4), 0, 15).to(torch.uint8)
    q_hi = q4 >> 2
    q_lo = q4 & 3

    # 无偏 2-bit Base 重建 (+1.5 中点补偿) 与 base + residual 无损 4-bit 重建
    x_base_hat = x_min + (4.0 * q_hi.float() + 1.5) * delta_4
    x_int4_hat = x_base_hat + (q_lo.float() - 1.5) * delta_4
    return q_hi, q_lo, x_min, delta_4, x_base_hat.to(x.dtype), x_int4_hat.to(x.dtype)


def quantize_nested_4bit_bitplane(
    x: torch.Tensor,
    reduce_dim: int = -1,
) -> dict[str, torch.Tensor]:
    """公开接口：对任意张量执行硬件原生 4-bit 嵌套双位面切分与双路径重建。"""
    q_hi, q_lo, x_min, delta_4, x_base_hat, x_int4_hat = _quantize_nested_tensors(x, reduce_dim)
    return {
        "q_hi": q_hi,
        "q_lo": q_lo,
        "x_min": x_min,
        "delta_4": delta_4,
        "x_base_hat": x_base_hat,
        "x_int4_hat": x_int4_hat,
    }


def quantize_uniform_2bit(
    x: torch.Tensor,
    reduce_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行标准独立 Uniform INT2（4 级均匀量化：0, 1, 2, 3）并反量化。"""
    x_fp32 = x.float()
    x_min = x_fp32.min(dim=reduce_dim, keepdim=True).values
    x_max = x_fp32.max(dim=reduce_dim, keepdim=True).values
    delta_2 = torch.clamp(x_max - x_min, min=1e-8) / 3.0
    q2 = torch.clamp(torch.round((x_fp32 - x_min) / delta_2), 0, 3).to(torch.uint8)
    x_hat = (x_min + q2.float() * delta_2).to(x.dtype)
    return q2, x_hat


def _build_quantized_pages(
    key: torch.Tensor,
    value: torch.Tensor,
    hist_len: int,
    page_size: int,
) -> QuantizedBitPlanePages:
    """
    为历史前缀 [0, hist_len) 构造页对齐的嵌套双位面与 Uniform INT2 表示。
      - Key：页内 per-channel (沿页内 token 维 dim=-2 量化)
      - Value：per-token (沿 head_dim 维 dim=-1 量化)
    """
    bsz, h_kv, _, head_dim = key.shape
    num_pages = hist_len // page_size

    k_hist = key[:, :, :hist_len, :]
    v_hist = value[:, :, :hist_len, :]

    # Key: (B, H_kv, P, S, d) 沿 S 维 (dim=-2) 做页内通道量化
    k_paged = k_hist.view(bsz, h_kv, num_pages, page_size, head_dim)
    _, _, _, _, k_base_p, k_int4_p = _quantize_nested_tensors(k_paged, reduce_dim=-2)
    _, k_int2_p = quantize_uniform_2bit(k_paged, reduce_dim=-2)

    k_page_min = k_paged.float().min(dim=-2).values
    k_page_max = k_paged.float().max(dim=-2).values

    # Value: (B, H_kv, hist_len, d) 沿 head_dim (dim=-1) 做 per-token 量化
    _, _, _, _, v_base, v_int4 = _quantize_nested_tensors(v_hist, reduce_dim=-1)
    _, v_int2 = quantize_uniform_2bit(v_hist, reduce_dim=-1)

    return QuantizedBitPlanePages(
        page_size=page_size,
        hist_len=hist_len,
        num_pages=num_pages,
        k_base=k_base_p.view(bsz, h_kv, hist_len, head_dim),
        k_int4=k_int4_p.view(bsz, h_kv, hist_len, head_dim),
        k_int2=k_int2_p.view(bsz, h_kv, hist_len, head_dim),
        v_base=v_base,
        v_int4=v_int4,
        v_int2=v_int2,
        k_page_min=k_page_min,
        k_page_max=k_page_max,
    )


def _repeat_kv_tensor(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """将 KV 头按 GQA 分组倍率展开为与 Q 头对齐的张量 (B, H_kv*grp, ...)。"""
    if num_groups == 1:
        return x
    bsz, h_kv, seq_len, head_dim = x.shape
    return x[:, :, None, :, :].expand(bsz, h_kv, num_groups, seq_len, head_dim).reshape(
        bsz, h_kv * num_groups, seq_len, head_dim
    )


def _compute_gqa_union_stats(selected_mask: torch.Tensor, num_kv: int, grp: int) -> tuple[float, float]:
    """
    统计每头独立选页导致的 GQA 组页并集膨胀情况。
    selected_mask 形状为 (B, H_q, L_q, P) bool。
    返回 (实际组内并集平均加载比例, 并集膨胀倍数 = union_ratio / per_head_ratio)。
    """
    bsz, _, lq, num_pages = selected_mask.shape
    if num_pages == 0:
        return 0.0, 1.0
    per_head_ratio = float(selected_mask.float().mean().item())
    grouped = selected_mask.view(bsz, num_kv, grp, lq, num_pages)
    union_mask = grouped.any(dim=2)
    union_ratio = float(union_mask.float().mean().item())
    inflation = union_ratio / max(per_head_ratio, 1e-12)
    return union_ratio, inflation


def paged_bitplane_attention_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
    mode: str = "bitplane",
    page_size: int = 32,
    recent_tokens: int = 32,
    hist_len: int | None = None,
    top_ratio: float = 0.15,
    routing: str = "page_max",
    gqa_shared: bool = True,
    pages_cache: QuantizedBitPlanePages | None = None,
    tau_hot: float = 4.0,
    tau_skip: float = 8.5,
    hot_ratio: float = 0.15,
    warm_ratio: float = 0.35,
    ellipsoid_beta: float = 1.5,
    pin_page0: bool = True,
    score_mode: str = "anisotropic",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    两阶段查询感知页级解码核心算子（支持全部对比基线与消融模式）。

    Args:
        query: 查询张量 (B, H_q, L_q, d)
        key: 完整键缓存 (B, H_kv, L_k, d)
        value: 完整值缓存 (B, H_kv, L_k, d)
        scaling: 注意力缩放因子（通常为 1 / sqrt(d)）
        attention_mask: 可选外部加性因果/填充掩码 (B, 1/H_q, L_q, L_k)
        mode: 解码模式，可选：
            - "fp16" / "bf16"：全精对照上限
            - "int4"：历史整页统一 4-bit
            - "int2"：历史整页统一标准 INT2
            - "base2"：历史整页统一使用无偏 2-bit Base 位面
            - "quest"：Quest Min-Max 估计选 Top-k 页读全精，其余页丢弃为 -inf
            - "bitplane"：2-bit Base 全扫打分 + Top-k 页恢复 4-bit + 未选中页保留 2-bit 长尾保底
            - "bitplane_gated"：单 Pass 静态锚点门控（Hot读4-bit / Warm读2-bit Base / Skip跳过V）
            - "bitplane_dualstream"：零额外元数据各向异性页排序启发式路由 + 零分支双流解码
            - "bitplane_droptail"：2-bit Base 选 Top-k 页恢复 4-bit，未选中页丢弃为 -inf（消融）
            - "key_only_int2"：历史 K 用 INT2，V 保持全精（诊断 K/V 敏感度）
            - "val_only_int2"：历史 K 保持全精，V 用 INT2（诊断 K/V 敏感度）
            - "bitplane_k4_v2"：历史 K 经 bitplane 路由后 Top-k 读 4-bit/其余 2-bit，V 统一 INT2
        page_size: PagedAttention 物理页大小，默认 32
        recent_tokens: 所有方法严格对称保护的尾部局部高精度最小窗口，默认 32
        hist_len: 可显式指定固定历史前缀页边界（须为 page_size 整数倍且 <= L_k - L_q）；
                  若为 None，则在严格因果安全区间内自动切分整页
        top_ratio: Stage 2 精修加载的页比例 rho，默认 0.15
        routing: Bit-Plane Stage 1 页路由打分准则 ("page_max" | "page_mass")
        gqa_shared: 是否按 GQA KV 组共享选出同一套 Top-k 页（True 消除组内膨胀）
        pages_cache: 可选预计算好的量化页对象以复用跨步解码开销
        score_mode: `bitplane_dualstream` 页排序策略 ("anisotropic" | "center" | "box_l1" | "ellipsoid")

    Returns:
        (attn_output, stats_dict)
        attn_output 形状为 (B, L_q, H_q, d)，与 transformers 期望输出布局一致。
    """
    bsz, h_q, lq, head_dim = query.shape
    _, h_kv, lk, _ = key.shape
    grp = h_q // h_kv
    neg_inf = torch.finfo(torch.float32).min

    # 构造严格因果掩码（针对 L_q > 1 时必须保证未来位置不可见）
    past_len = lk - lq
    q_pos = torch.arange(past_len, lk, device=query.device)[:, None]
    k_pos = torch.arange(lk, device=query.device)[None, :]
    base_causal = torch.where(k_pos <= q_pos, 0.0, neg_inf).to(torch.float32)[None, None, :, :]
    if attention_mask is not None:
        # 外部 mask 若为 4D 加性掩码，与严格因果上界取较小值确保零泄露
        ext_mask = attention_mask.to(torch.float32)
        if ext_mask.shape[-2] == lq and ext_mask.shape[-1] == lk:
            base_causal = torch.minimum(base_causal, ext_mask)

    # 计算严格安全的前缀整页边界 hist_len：
    # 必须同时保证：
    #   1) 末尾至少留足 recent_tokens 个 token 处于全精保护；
    #   2) hist_len <= past_len（若在连续 chunk 内，所有历史页均早于 chunk 起点，消除跨步依赖）。
    if hist_len is None:
        if lq == 1:
            safe_upper = max(0, lk - recent_tokens)
        else:
            safe_upper = max(0, min(past_len, lk - recent_tokens))
        hist_len_eff = (safe_upper // page_size) * page_size
    else:
        hist_len_eff = (max(0, min(int(hist_len), past_len if lq > 1 else lk)) // page_size) * page_size

    num_pages = hist_len_eff // page_size

    # 若 mode == "fp16"/"bf16" 或无完整历史页，直接走原生全精注意力路径
    if mode in ("fp16", "bf16") or num_pages == 0:
        k_full = _repeat_kv_tensor(key, grp).float()
        v_full = _repeat_kv_tensor(value, grp).float()
        q_fp32 = query.float()
        logits = torch.matmul(q_fp32, k_full.transpose(-2, -1)) * scaling + base_causal
        probs = torch.softmax(logits, dim=-1)
        out = torch.matmul(probs, v_full).transpose(1, 2).contiguous().to(query.dtype)
        return out, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "selected_ratio": 1.0 if mode in ("fp16", "bf16") else 0.0,
            "union_ratio": 1.0 if mode in ("fp16", "bf16") else 0.0,
            "union_inflation": 1.0,
            "eff_bits_per_elem": 16.0,
            "logical_read_bits_per_kv_elem": 16.0,
        }

    # 构建或复用历史整页量化表（纯前缀只读）
    if pages_cache is not None and pages_cache.hist_len == hist_len_eff and pages_cache.page_size == page_size:
        pages = pages_cache
    else:
        pages = _build_quantized_pages(key, value, hist_len_eff, page_size)

    q_fp32 = query.float()
    k_tail_full = _repeat_kv_tensor(key[:, :, hist_len_eff:, :], grp).float()
    v_tail_full = _repeat_kv_tensor(value[:, :, hist_len_eff:, :], grp).float()
    mask_hist = base_causal[:, :, :, :hist_len_eff]
    mask_tail = base_causal[:, :, :, hist_len_eff:]

    def _finish_attention(
        k_hist_eff: torch.Tensor,
        v_hist_eff: torch.Tensor,
        hist_mask_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """统一拼接历史区与全精尾部区并计算因果 Softmax 输出。"""
        k_h = _repeat_kv_tensor(k_hist_eff, grp).float() if k_hist_eff.shape[1] == h_kv else k_hist_eff.float()
        v_h = _repeat_kv_tensor(v_hist_eff, grp).float() if v_hist_eff.shape[1] == h_kv else v_hist_eff.float()

        logits_h = torch.matmul(q_fp32, k_h.transpose(-2, -1)) * scaling
        logits_h = logits_h + (hist_mask_override if hist_mask_override is not None else mask_hist)
        logits_t = torch.matmul(q_fp32, k_tail_full.transpose(-2, -1)) * scaling + mask_tail

        logits_all = torch.cat([logits_h, logits_t], dim=-1)
        probs = torch.softmax(logits_all, dim=-1)
        p_h = probs[:, :, :, :hist_len_eff]
        p_t = probs[:, :, :, hist_len_eff:]
        out_fp32 = torch.matmul(p_h, v_h) + torch.matmul(p_t, v_tail_full)
        return out_fp32.transpose(1, 2).contiguous().to(query.dtype)

    # 1) 全历史统一量化静态模式：int4 / int2 / base2 / key_only_int2 / val_only_int2
    if mode == "int4":
        out = _finish_attention(pages.k_int4, pages.v_int4)
        lrb = compute_logical_read_bits_per_kv_elem("int4", head_dim=head_dim, page_size=page_size)
        return out, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "selected_ratio": 1.0,
            "union_ratio": 1.0,
            "union_inflation": 1.0,
            "eff_bits_per_elem": lrb,
            "logical_read_bits_per_kv_elem": lrb,
        }
    if mode == "int2":
        out = _finish_attention(pages.k_int2, pages.v_int2)
        lrb = compute_logical_read_bits_per_kv_elem("int2", head_dim=head_dim, page_size=page_size)
        return out, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "selected_ratio": 0.0,
            "union_ratio": 0.0,
            "union_inflation": 1.0,
            "eff_bits_per_elem": lrb,
            "logical_read_bits_per_kv_elem": lrb,
        }
    if mode == "base2":
        out = _finish_attention(pages.k_base, pages.v_base)
        lrb = compute_logical_read_bits_per_kv_elem("base2", head_dim=head_dim, page_size=page_size)
        return out, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "selected_ratio": 0.0,
            "union_ratio": 0.0,
            "union_inflation": 1.0,
            "eff_bits_per_elem": lrb,
            "logical_read_bits_per_kv_elem": lrb,
        }
    if mode == "key_only_int2":
        out = _finish_attention(pages.k_int2, value[:, :, :hist_len_eff, :])
        lrb = compute_logical_read_bits_per_kv_elem("key_only_int2", head_dim=head_dim, page_size=page_size)
        return out, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "selected_ratio": 0.0,
            "union_ratio": 0.0,
            "union_inflation": 1.0,
            "eff_bits_per_elem": lrb,
            "logical_read_bits_per_kv_elem": lrb,
        }
    if mode == "val_only_int2":
        out = _finish_attention(key[:, :, :hist_len_eff, :], pages.v_int2)
        lrb = compute_logical_read_bits_per_kv_elem("val_only_int2", head_dim=head_dim, page_size=page_size)
        return out, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "selected_ratio": 0.0,
            "union_ratio": 0.0,
            "union_inflation": 1.0,
            "eff_bits_per_elem": lrb,
            "logical_read_bits_per_kv_elem": lrb,
        }

    # 计算当前配额需要选中的历史页数 top_k
    if top_ratio <= 0.0:
        top_k = 0
    elif top_ratio >= 1.0:
        top_k = num_pages
    else:
        top_k = max(1, min(num_pages, int(math.ceil(num_pages * top_ratio))))
    actual_top_ratio = top_k / num_pages

    # 2) Quest 基线（页 Min-Max 估计选 Top-k 页读取全精 KV，其余页直接丢弃）
    if mode == "quest":
        # q_g: (B, H_kv, grp, L_q, d)
        q_g = q_fp32.view(bsz, h_kv, grp, lq, head_dim)
        # k_min / k_max: (B, H_kv, 1, 1, P, d) vs q_g (..., L_q, 1, d)
        kmin_e = pages.k_page_min[:, :, None, None, :, :]
        kmax_e = pages.k_page_max[:, :, None, None, :, :]
        q_e = q_g.unsqueeze(-2)  # (B, H_kv, grp, L_q, 1, d)
        # Quest 官方页得分估计：sum_c max(q_c * min_c, q_c * max_c) * scaling
        q_min = q_e * kmin_e
        q_max = q_e * kmax_e
        page_scores_g = torch.maximum(q_min, q_max).sum(dim=-1) * scaling  # (B, H_kv, grp, L_q, P)

        if gqa_shared:
            # 将组内各头相对最大值居中后相加（保证各头同量纲竞争且完全共享 KV 页）
            head_rel = page_scores_g - page_scores_g.max(dim=-1, keepdim=True).values
            grp_scores = torch.exp(head_rel).sum(dim=2)  # (B, H_kv, L_q, P)
            sel_mask_g = torch.zeros((bsz, h_kv, lq, num_pages), device=query.device, dtype=torch.bool)
            if top_k > 0:
                topk_idx = torch.topk(grp_scores, k=top_k, dim=-1).indices  # (B, H_kv, L_q, top_k)
                sel_mask_g.scatter_(-1, topk_idx, True)
            sel_mask_hq = sel_mask_g[:, :, None, :, :].expand(bsz, h_kv, grp, lq, num_pages).reshape(
                bsz, h_q, lq, num_pages
            )
        else:
            page_scores_hq = page_scores_g.view(bsz, h_q, lq, num_pages)
            sel_mask_hq = torch.zeros((bsz, h_q, lq, num_pages), device=query.device, dtype=torch.bool)
            if top_k > 0:
                topk_idx = torch.topk(page_scores_hq, k=top_k, dim=-1).indices
                sel_mask_hq.scatter_(-1, topk_idx, True)

        union_ratio, inflation = _compute_gqa_union_stats(sel_mask_hq, h_kv, grp)

        # 未选中的历史页丢弃为 -inf；选中页使用全精 KV
        tok_keep_mask = sel_mask_hq[:, :, :, :, None].expand(bsz, h_q, lq, num_pages, page_size).reshape(
            bsz, h_q, lq, hist_len_eff
        )
        quest_hist_mask = torch.where(tok_keep_mask, mask_hist, torch.full_like(mask_hist, neg_inf))
        out = _finish_attention(key[:, :, :hist_len_eff, :], value[:, :, :hist_len_eff, :], quest_hist_mask)
        lrb = compute_logical_read_bits_per_kv_elem("quest", hot_ratio=union_ratio, head_dim=head_dim, page_size=page_size)
        return out, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "selected_ratio": actual_top_ratio,
            "union_ratio": union_ratio,
            "union_inflation": inflation,
            "eff_bits_per_elem": lrb,
            "logical_read_bits_per_kv_elem": lrb,
        }

    # 3) Single-Pass Anchor-Gated Bit-Plane 模式
    if mode == "bitplane_gated":
        k_base_hq = _repeat_kv_tensor(pages.k_base, grp).float()
        v_base_hq = _repeat_kv_tensor(pages.v_base, grp).float()
        k_int4_hq = _repeat_kv_tensor(pages.k_int4, grp).float()
        v_int4_hq = _repeat_kv_tensor(pages.v_int4, grp).float()

        logits_base_hist = torch.matmul(q_fp32, k_base_hq.transpose(-2, -1)) * scaling + mask_hist
        logits_tail = torch.matmul(q_fp32, k_tail_full.transpose(-2, -1)) * scaling + mask_tail

        # 固定锚点 m_anchor = max(Page 0 2-bit Base 最大 logit, 全精尾部最大 logit)
        m_p0 = logits_base_hist[:, :, :, :page_size].max(dim=-1).values
        if k_tail_full.shape[2] > 0:
            m_tail = logits_tail.max(dim=-1).values
            m_anchor = torch.maximum(m_p0, m_tail)
        else:
            m_anchor = m_p0

        # 页级相对差 Δ_p = max_{h in G_g} (m_page[h] - m_anchor[h])，按 GQA 组共享判决
        logits_base_paged = logits_base_hist.view(bsz, h_kv, grp, lq, num_pages, page_size)
        m_page = logits_base_paged.max(dim=-1).values  # (B, H_kv, grp, L_q, P)
        m_anchor_g = m_anchor.view(bsz, h_kv, grp, lq, 1)
        delta_page = (m_page - m_anchor_g).max(dim=2).values  # (B, H_kv, L_q, P)

        # 第 0 页 (Attention Sink) 强制纳入 Hot 路径
        is_hot_g = delta_page > (-tau_hot)
        is_hot_g[:, :, :, 0] = True
        is_skip_g = delta_page <= (-tau_skip)
        is_skip_g[:, :, :, 0] = False
        is_warm_g = (~is_hot_g) & (~is_skip_g)

        hot_ratio = float(is_hot_g.float().mean().item())
        warm_ratio = float(is_warm_g.float().mean().item())
        skip_ratio = float(is_skip_g.float().mean().item())
        lrb = compute_logical_read_bits_per_kv_elem(
            "bitplane_gated", hot_ratio=hot_ratio, warm_ratio=warm_ratio, skip_ratio=skip_ratio,
            head_dim=head_dim, page_size=page_size,
        )

        def _expand_to_tokens(mask_g: torch.Tensor) -> torch.Tensor:
            return (
                mask_g[:, :, None, :, :, None]
                .expand(bsz, h_kv, grp, lq, num_pages, page_size)
                .reshape(bsz, h_q, lq, hist_len_eff)
            )

        tok_hot = _expand_to_tokens(is_hot_g)
        tok_skip = _expand_to_tokens(is_skip_g)

        logits_int4_hist = torch.matmul(q_fp32, k_int4_hq.transpose(-2, -1)) * scaling + mask_hist
        logits_hist_eff = torch.where(tok_hot, logits_int4_hist, logits_base_hist)
        logits_hist_eff = torch.where(tok_skip, torch.full_like(logits_hist_eff, neg_inf), logits_hist_eff)

        logits_final = torch.cat([logits_hist_eff, logits_tail], dim=-1)
        probs_final = torch.softmax(logits_final, dim=-1)
        p_hist = probs_final[:, :, :, :hist_len_eff]
        p_tail = probs_final[:, :, :, hist_len_eff:]

        p_hot = torch.where(tok_hot, p_hist, torch.zeros_like(p_hist))
        p_warm = torch.where((~tok_hot) & (~tok_skip), p_hist, torch.zeros_like(p_hist))
        out_hist = torch.matmul(p_hot, v_int4_hq) + torch.matmul(p_warm, v_base_hq)
        out_total = (out_hist + torch.matmul(p_tail, v_tail_full)).transpose(1, 2).contiguous().to(query.dtype)
        return out_total, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "hot_ratio": hot_ratio,
            "warm_ratio": warm_ratio,
            "skip_ratio": skip_ratio,
            "phys_read_bits": lrb,
            "logical_read_bits_per_kv_elem": lrb,
            "selected_ratio": hot_ratio,
            "union_ratio": hot_ratio,
            "union_inflation": 1.0,
            "eff_bits_per_elem": lrb,
        }

    # 3B) 各向异性页排序启发式路由双流 Bit-Plane 模式（零额外元数据路由 + 零分支双流语义）
    if mode == "bitplane_dualstream":
        if hot_ratio <= 0.0:
            n_hot = 0
        else:
            n_hot = max(1, min(num_pages, int(math.ceil(num_pages * hot_ratio))))
        if warm_ratio <= 0.0:
            n_warm = 0
        else:
            n_warm = max(0, min(num_pages - n_hot, int(math.ceil(num_pages * warm_ratio))))
        n_keep = n_hot + n_warm
        if n_keep <= 0:
            raise ValueError("hot_ratio 与 warm_ratio 不能同时为 0")

        # 复用已有页元数据 (k_min, k_scale) 构造页中心与半宽（零额外索引内存开销）
        k_min_p = pages.k_page_min.float()
        k_scale_p = torch.clamp(pages.k_page_max.float() - k_min_p, min=0.0) / 15.0
        k_mid = k_min_p + 7.5 * k_scale_p
        k_rad = 7.5 * k_scale_p

        q_g = q_fp32.view(bsz, h_kv, grp, lq, head_dim) * scaling
        mu = torch.einsum("bhgtd,bhpd->bhgtp", q_g, k_mid)
        s_mode = score_mode.lower()
        if s_mode == "center" or (s_mode in ("anisotropic", "ellipsoid") and float(ellipsoid_beta) <= 0.0):
            u = mu
        elif s_mode in ("anisotropic", "ellipsoid"):
            var = torch.einsum("bhgtd,bhpd->bhgtp", q_g * q_g, k_rad * k_rad)
            u = mu + ellipsoid_beta * torch.sqrt(torch.clamp(var, min=0.0))
        elif s_mode == "box_l1":
            u = mu + torch.einsum("bhgtd,bhpd->bhgtp", torch.abs(q_g), k_rad)
        else:
            raise ValueError(f"不支持的 score_mode: {score_mode}")

        u_rel = u - u.max(dim=-1, keepdim=True).values
        # 使用极小量 GQA 组内概率密度和作为次级确定性平局消解项，避免多头各排第一时 max 出现 0.0 平局
        grp_scores = u_rel.max(dim=2).values + 1e-5 * torch.exp(u_rel).sum(dim=2)  # (B, H_kv, L_q, P)
        if pin_page0 and num_pages > 0:
            grp_scores = grp_scores.clone()
            grp_scores[:, :, :, 0] = 1e9

        is_hot_g = torch.zeros((bsz, h_kv, lq, num_pages), device=query.device, dtype=torch.bool)
        is_warm_g = torch.zeros((bsz, h_kv, lq, num_pages), device=query.device, dtype=torch.bool)
        top_idx = torch.topk(grp_scores, k=n_keep, dim=-1, sorted=True).indices
        if n_hot > 0:
            is_hot_g.scatter_(-1, top_idx[..., :n_hot], True)
        if n_warm > 0:
            is_warm_g.scatter_(-1, top_idx[..., n_hot:n_keep], True)
        is_skip_g = (~is_hot_g) & (~is_warm_g)

        def _expand_dual(mask_g: torch.Tensor) -> torch.Tensor:
            return (
                mask_g[:, :, None, :, :, None]
                .expand(bsz, h_kv, grp, lq, num_pages, page_size)
                .reshape(bsz, h_q, lq, hist_len_eff)
            )

        tok_hot = _expand_dual(is_hot_g)
        tok_warm = _expand_dual(is_warm_g)
        tok_skip = _expand_dual(is_skip_g)

        k_base_hq = _repeat_kv_tensor(pages.k_base, grp).float()
        v_base_hq = _repeat_kv_tensor(pages.v_base, grp).float()
        k_int4_hq = _repeat_kv_tensor(pages.k_int4, grp).float()
        v_int4_hq = _repeat_kv_tensor(pages.v_int4, grp).float()

        logits_base_hist = torch.matmul(q_fp32, k_base_hq.transpose(-2, -1)) * scaling + mask_hist
        logits_int4_hist = torch.matmul(q_fp32, k_int4_hq.transpose(-2, -1)) * scaling + mask_hist
        logits_tail = torch.matmul(q_fp32, k_tail_full.transpose(-2, -1)) * scaling + mask_tail

        logits_hist_eff = torch.where(tok_hot, logits_int4_hist, logits_base_hist)
        logits_hist_eff = torch.where(tok_skip, torch.full_like(logits_hist_eff, neg_inf), logits_hist_eff)

        logits_final = torch.cat([logits_hist_eff, logits_tail], dim=-1)
        probs_final = torch.softmax(logits_final, dim=-1)
        p_hist = probs_final[:, :, :, :hist_len_eff]
        p_tail = probs_final[:, :, :, hist_len_eff:]

        p_hot = torch.where(tok_hot, p_hist, torch.zeros_like(p_hist))
        p_warm = torch.where(tok_warm, p_hist, torch.zeros_like(p_hist))
        out_hist = torch.matmul(p_hot, v_int4_hq) + torch.matmul(p_warm, v_base_hq)
        out_total = (out_hist + torch.matmul(p_tail, v_tail_full)).transpose(1, 2).contiguous().to(query.dtype)

        actual_hot_r = n_hot / num_pages
        actual_warm_r = n_warm / num_pages
        actual_skip_r = max(0.0, 1.0 - actual_hot_r - actual_warm_r)
        if n_hot == num_pages:
            lrb = compute_logical_read_bits_per_kv_elem("int4", head_dim=head_dim, page_size=page_size)
        else:
            lrb = compute_logical_read_bits_per_kv_elem(
                "bitplane_dualstream", hot_ratio=actual_hot_r, warm_ratio=actual_warm_r, skip_ratio=actual_skip_r,
                head_dim=head_dim, page_size=page_size,
            )
        return out_total, {
            "hist_len": hist_len_eff,
            "num_pages": num_pages,
            "hot_ratio": actual_hot_r,
            "warm_ratio": actual_warm_r,
            "skip_ratio": actual_skip_r,
            "phys_read_bits": lrb,
            "logical_read_bits_per_kv_elem": lrb,
            "selected_ratio": actual_hot_r,
            "union_ratio": actual_hot_r,
            "union_inflation": 1.0,
            "eff_bits_per_elem": lrb,
        }

    # 4) Bit-Plane 主方法与核心消融：bitplane / bitplane_droptail / bitplane_k4_v2
    if mode not in ("bitplane", "bitplane_droptail", "bitplane_k4_v2"):
        raise ValueError(f"不支持的解码模式: {mode}")

    # Stage 1：全扫描 2-bit K_base 计算所有历史页的粗 logits，零额外路由元数据！
    k_base_hq = _repeat_kv_tensor(pages.k_base, grp).float()
    v_base_hq = _repeat_kv_tensor(pages.v_base, grp).float()
    k_int4_hq = _repeat_kv_tensor(pages.k_int4, grp).float()
    v_int4_hq = _repeat_kv_tensor(pages.v_int4, grp).float()

    logits_base_hist = torch.matmul(q_fp32, k_base_hq.transpose(-2, -1)) * scaling + mask_hist
    logits_tail = torch.matmul(q_fp32, k_tail_full.transpose(-2, -1)) * scaling + mask_tail

    # 计算页路由分数：先算出 2-bit Base 在当前全可见区上对应的归一化注意力质量（跨 GQA 头可比）
    logits_base_all = torch.cat([logits_base_hist, logits_tail], dim=-1)
    probs_base_hist = torch.softmax(logits_base_all, dim=-1)[:, :, :, :hist_len_eff]
    probs_base_paged = probs_base_hist.view(bsz, h_kv, grp, lq, num_pages, page_size)

    if routing == "page_max":
        # 每头每页取最大单针概率，并在 GQA 组内按组求和或取 max（取 max 确保组内任意头发现单针即召回）
        head_page_scores = probs_base_paged.max(dim=-1).values  # (B, H_kv, grp, L_q, P)
        grp_page_scores = head_page_scores.max(dim=2).values  # (B, H_kv, L_q, P)
    elif routing == "page_mass":
        # 每页概率质量求和，GQA 组内加总
        head_page_scores = probs_base_paged.sum(dim=-1)  # (B, H_kv, grp, L_q, P)
        grp_page_scores = head_page_scores.sum(dim=2)  # (B, H_kv, L_q, P)
    else:
        raise ValueError(f"不支持的 routing 策略: {routing}")

    # 选页（支持 GQA 共享或逐头）
    if gqa_shared:
        sel_mask_g = torch.zeros((bsz, h_kv, lq, num_pages), device=query.device, dtype=torch.bool)
        if top_k > 0:
            topk_idx = torch.topk(grp_page_scores, k=top_k, dim=-1).indices
            sel_mask_g.scatter_(-1, topk_idx, True)
        sel_mask_hq = sel_mask_g[:, :, None, :, :].expand(bsz, h_kv, grp, lq, num_pages).reshape(
            bsz, h_q, lq, num_pages
        )
    else:
        head_only_scores = head_page_scores.view(bsz, h_q, lq, num_pages)
        sel_mask_hq = torch.zeros((bsz, h_q, lq, num_pages), device=query.device, dtype=torch.bool)
        if top_k > 0:
            topk_idx = torch.topk(head_only_scores, k=top_k, dim=-1).indices
            sel_mask_hq.scatter_(-1, topk_idx, True)

    union_ratio, inflation = _compute_gqa_union_stats(sel_mask_hq, h_kv, grp)
    tok_sel_mask = sel_mask_hq[:, :, :, :, None].expand(bsz, h_q, lq, num_pages, page_size).reshape(
        bsz, h_q, lq, hist_len_eff
    )

    # Stage 2：对入选 Top-k 热点页计算 4-bit 精修 logits；未入选页保留 2-bit Base 或丢弃
    logits_int4_hist = torch.matmul(q_fp32, k_int4_hq.transpose(-2, -1)) * scaling + mask_hist
    if mode == "bitplane_droptail":
        logits_hist_eff = torch.where(tok_sel_mask, logits_int4_hist, torch.full_like(logits_int4_hist, neg_inf))
    else:
        logits_hist_eff = torch.where(tok_sel_mask, logits_int4_hist, logits_base_hist)

    logits_final = torch.cat([logits_hist_eff, logits_tail], dim=-1)
    probs_final = torch.softmax(logits_final, dim=-1)
    p_hist = probs_final[:, :, :, :hist_len_eff]
    p_tail = probs_final[:, :, :, hist_len_eff:]

    # 对 Value 做对应两阶段加权求和：入选页用 V_int4，未入选长尾页用 V_base（或 INT2 消融）
    if mode == "bitplane_k4_v2":
        v_int2_hq = _repeat_kv_tensor(pages.v_int2, grp).float()
        out_hist = torch.matmul(p_hist, v_int2_hq)
        eff_bits = compute_logical_read_bits_per_kv_elem("bitplane_k4_v2", hot_ratio=union_ratio, head_dim=head_dim, page_size=page_size)
    elif mode == "bitplane_droptail":
        # 仅选中页参与输出；未选中页 p_hist 已接近 0
        out_hist = torch.matmul(p_hist, v_int4_hq)
        eff_bits = compute_logical_read_bits_per_kv_elem("bitplane_droptail", hot_ratio=union_ratio, head_dim=head_dim, page_size=page_size)
    else:
        # 标准 Bit-Plane：入选页概率质量走 V_int4，未入选长尾页概率质量走 V_base
        p_sel = torch.where(tok_sel_mask, p_hist, torch.zeros_like(p_hist))
        p_unsel = torch.where(tok_sel_mask, torch.zeros_like(p_hist), p_hist)
        out_hist = torch.matmul(p_sel, v_int4_hq) + torch.matmul(p_unsel, v_base_hq)
        eff_bits = compute_logical_read_bits_per_kv_elem("bitplane", hot_ratio=union_ratio, head_dim=head_dim, page_size=page_size)

    out_total = (out_hist + torch.matmul(p_tail, v_tail_full)).transpose(1, 2).contiguous().to(query.dtype)
    return out_total, {
        "hist_len": hist_len_eff,
        "num_pages": num_pages,
        "selected_ratio": actual_top_ratio,
        "union_ratio": union_ratio,
        "union_inflation": inflation,
        "eff_bits_per_elem": eff_bits,
        "logical_read_bits_per_kv_elem": eff_bits,
    }


def compute_layer_outlier_stats(
    key: torch.Tensor,
    value: torch.Tensor,
    page_size: int = 32,
) -> dict[str, float]:
    """
    统计单层真实 K/V 缓存的离群尖峰特征与页内 2-bit 相对重建误差。

    Args:
        key: (B, H_kv, L, d)
        value: (B, H_kv, L, d)
        page_size: 分页大小

    Returns:
        包含 K/V 峰值-RMS 比 (max/rms)、通道 RMS 方差集中度（Max/Mean 通道比）、
        以及实际页粒度下 Uniform INT2 相对 L2 重建误差的字典。
    """
    k_fp = key[0].float()  # (H_kv, L, d)
    v_fp = value[0].float()
    h_kv, seq_len, head_dim = k_fp.shape
    hist_len = (seq_len // page_size) * page_size

    # 1) 全局离群峰值比：max(|x|) / rms(x)
    k_rms = float(torch.sqrt(torch.mean(k_fp ** 2)).item()) + 1e-12
    v_rms = float(torch.sqrt(torch.mean(v_fp ** 2)).item()) + 1e-12
    k_peak_ratio = float(k_fp.abs().max().item()) / k_rms
    v_peak_ratio = float(v_fp.abs().max().item()) / v_rms

    # 2) 通道维集中度：跨序列求每个通道能量 (H_kv, d)，比较最大通道 RMS 与中位通道 RMS
    k_ch_rms = torch.sqrt(torch.mean(k_fp ** 2, dim=1))  # (H_kv, d)
    v_ch_rms = torch.sqrt(torch.mean(v_fp ** 2, dim=1))
    k_ch_conc = float((k_ch_rms.max(dim=-1).values / (k_ch_rms.median(dim=-1).values + 1e-12)).mean().item())
    v_ch_conc = float((v_ch_rms.max(dim=-1).values / (v_ch_rms.median(dim=-1).values + 1e-12)).mean().item())

    # 3) 实际页面粒度下的 Uniform INT2 与 Nested INT4 相对 L2 误差
    if hist_len > 0:
        pages = _build_quantized_pages(key[:, :, :hist_len, :], value[:, :, :hist_len, :], hist_len, page_size)
        k_orig = key[:, :, :hist_len, :].float()
        v_orig = value[:, :, :hist_len, :].float()
        k_int2_rel = float((torch.norm(k_orig - pages.k_int2.float()) / (torch.norm(k_orig) + 1e-12)).item())
        k_base2_rel = float((torch.norm(k_orig - pages.k_base.float()) / (torch.norm(k_orig) + 1e-12)).item())
        k_int4_rel = float((torch.norm(k_orig - pages.k_int4.float()) / (torch.norm(k_orig) + 1e-12)).item())
        v_int2_rel = float((torch.norm(v_orig - pages.v_int2.float()) / (torch.norm(v_orig) + 1e-12)).item())
        v_base2_rel = float((torch.norm(v_orig - pages.v_base.float()) / (torch.norm(v_orig) + 1e-12)).item())
        v_int4_rel = float((torch.norm(v_orig - pages.v_int4.float()) / (torch.norm(v_orig) + 1e-12)).item())
    else:
        k_int2_rel = k_base2_rel = k_int4_rel = v_int2_rel = v_base2_rel = v_int4_rel = 0.0

    return {
        "k_peak_over_rms": k_peak_ratio,
        "v_peak_over_rms": v_peak_ratio,
        "k_channel_max_over_median": k_ch_conc,
        "v_channel_max_over_median": v_ch_conc,
        "k_int2_rel_err": k_int2_rel,
        "k_base2_rel_err": k_base2_rel,
        "k_int4_rel_err": k_int4_rel,
        "v_int2_rel_err": v_int2_rel,
        "v_base2_rel_err": v_base2_rel,
        "v_int4_rel_err": v_int4_rel,
    }
