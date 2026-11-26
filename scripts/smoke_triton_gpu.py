#!/usr/bin/env python3
"""
RTX 5070 Ti (Blackwell) GPU + Triton 编译与位面解包点积冒烟验证脚本。

验证内容：
  1. 打印 torch / CUDA / GPU 型号与算力 / triton / transformers 版本信息；
  2. 编译运行最小 @triton.jit 核函数 `bitplane_unpack_dot_kernel`，验证 GPU 上
     2-bit Base + 2-bit Residual 位面解包重组与点积精度（max_abs_err < 1e-4）；
  3. 验证只读加载本地 Qwen3-0.6B 与 Qwen3.5-0.8B 的配置与分词器。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
import transformers
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_model_source(model_name: str) -> str:
    """优先读取 BITPLANE_MODEL_CACHE 或本地 .model_cache，否则回退至 HuggingFace Hub。"""
    env_root = os.environ.get("BITPLANE_MODEL_CACHE")
    if env_root:
        cand = Path(env_root) / model_name
        if cand.exists():
            return str(cand)
    local_cand = PROJECT_ROOT / ".model_cache" / model_name
    if local_cand.exists():
        return str(local_cand)
    return f"Qwen/{model_name}"


@triton.jit
def bitplane_unpack_dot_kernel(
    q_ptr,
    q_hi_ptr,
    q_lo_ptr,
    x_min_ptr,
    scale_ptr,
    out_ptr,
    N_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    在 Triton 核内将打包的 uint8 双位面 (q_hi, q_lo) 解包并恢复 4-bit 向量，与 query 计算点积。

    每个 program 处理一个 token 向量（长度 HEAD_DIM）：
      x_hat = x_min + (4 * q_hi + q_lo) * scale
      out[tok] = sum_d q[d] * x_hat[d]
    """
    tok_idx = tl.program_id(0)
    if tok_idx >= N_TOKENS:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    # 读取 query（1D 向量）、位面整数及量化尺度
    q_vec = tl.load(q_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    base_offset = tok_idx * HEAD_DIM + offs_d
    q_hi = tl.load(q_hi_ptr + base_offset, mask=mask_d, other=0).to(tl.float32)
    q_lo = tl.load(q_lo_ptr + base_offset, mask=mask_d, other=0).to(tl.float32)

    x_min = tl.load(x_min_ptr + tok_idx).to(tl.float32)
    scale = tl.load(scale_ptr + tok_idx).to(tl.float32)

    # 位面组合重建 4-bit 量化值
    q4 = 4.0 * q_hi + q_lo
    x_hat = x_min + q4 * scale
    dot_val = tl.sum(q_vec * x_hat, axis=0)
    tl.store(out_ptr + tok_idx, dot_val)


def main() -> int:
    print("=== 环境与硬件信息 ===")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA 可用:     {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA 设备！")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU 型号:      {props.name}")
    print(f"Compute Cap:  sm_{props.major}{props.minor}")
    print(f"总显存:        {props.total_memory / (1024 ** 3):.2f} GB")
    print(f"Triton:       {triton.__version__}")
    print(f"Transformers: {transformers.__version__}")

    # 验证 Triton 编译与位面解包点积核
    print("\n=== Triton 位面解包点积核验证 ===")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    n_tokens = 256
    head_dim = 128
    block_d = triton.next_power_of_2(head_dim)

    q = torch.randn((head_dim,), device=device, dtype=torch.float32)
    x = torch.randn((n_tokens, head_dim), device=device, dtype=torch.float32) * 1.5

    x_min = x.min(dim=-1).values
    x_max = x.max(dim=-1).values
    scale = torch.clamp(x_max - x_min, min=1e-8) / 15.0
    q4 = torch.clamp(torch.round((x - x_min[:, None]) / scale[:, None]), 0, 15).to(torch.uint8)
    q_hi = (q4 >> 2).contiguous()
    q_lo = (q4 & 3).contiguous()

    # PyTorch float32 参考结果
    x_hat_ref = x_min[:, None] + (4.0 * q_hi.float() + q_lo.float()) * scale[:, None]
    out_ref = torch.matmul(x_hat_ref, q)

    # Triton 核执行结果
    out_triton = torch.empty((n_tokens,), device=device, dtype=torch.float32)
    grid = (n_tokens,)
    bitplane_unpack_dot_kernel[grid](
        q,
        q_hi,
        q_lo,
        x_min,
        scale,
        out_triton,
        N_TOKENS=n_tokens,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
    )
    torch.cuda.synchronize()

    max_abs_err = float((out_triton - out_ref).abs().max().item())
    mean_abs_err = float((out_triton - out_ref).abs().mean().item())
    print(f"max_abs_err:  {max_abs_err:.6e}")
    print(f"mean_abs_err: {mean_abs_err:.6e}")
    assert max_abs_err < 1e-4, f"Triton 解包点积误差超限: {max_abs_err}"

    # 验证模型配置与分词器加载（优先本地缓存，未配置时回退 HuggingFace Hub）
    print("\n=== 模型配置与分词器验证 ===")
    for name in ("Qwen3-0.6B", "Qwen3.5-0.8B"):
        m_src = _resolve_model_source(name)
        cfg = AutoConfig.from_pretrained(m_src)
        tok = AutoTokenizer.from_pretrained(m_src)
        sample_ids = tok.encode("Bit-plane paged attention smoke check.", add_special_tokens=False)
        print(f"{name} ({m_src}): model_type={cfg.model_type}, vocab={len(tok)}, sample_tokens={len(sample_ids)}")

    print("\n[PASS] RTX 5070 Ti GPU + Triton + 本地模型缓存冒烟验证全部通过！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
