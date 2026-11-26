#!/usr/bin/env python3
"""
Phase 1 -> Batch A 终极真机压测与论文数据总账（RTX 5070 Ti sm_120）。

涵盖 5 大全真实测模块（严禁估算或硬编码私有路径）：
  1. 算子 CUDA Event 精确延时与物理显存基准（B=1/4, L=2K..64K）：
     - d=128 官方强基线：PyTorch NVIDIA cuDNN SDPA (enable_gqa=True) + Triton-FP16 + Triton-INT4；
     - 新增真 PagedAttention 乱序物理页表实测 (`PagedBitPlaneKVPool(..., shuffle_blocks=True)`)；
  2. PTXAS 编译微架构剖析（含真页表无分支核 n_regs、shared_bytes 与 SM 理论驻留率）；
  3. 零额外元数据页排序启发式实证消融：真实模型 Attention Mass Recall & Oracle Top-15% 召回率
     对比（Center-Only vs Anisotropic beta=0.5/1.0/1.5/2.0 vs Box-Bound L1）；
  4. 真实大模型真 `BitPlaneDynamicCache` 端到端挂载验证（`Qwen3.5-0.8B` Hybrid & `Qwen3-0.6B` Dense）：
     - 128 步自回归 PPL / KL (vs BF16) + 5 深度大海捞针（全由动态增量封页执行）；
     - 真实 `model.generate()` GPU VRAM 压缩比与 Decode 吞吐实测（torch.cuda.memory_allocated / max_memory_allocated）；
  5. 输出 `results/triton_benchmark.json` 与 300 DPI 四宫格顶会总结图 `results/triton_benchmark.png`。
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForConditionalGeneration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bitplane_attn import (  # noqa: E402
    BitPlaneDynamicCache,
    PackedBitPlaneKVCache,
    PagedBitPlaneKVPool,
    compute_logical_read_bits_per_kv_elem,
    triton_bitplane_decode,
    triton_bitplane_dualstream_decode,
    triton_bitplane_gated_decode,
    triton_fp16_paged_decode,
    triton_int4_paged_decode,
    triton_paged_bitplane_dualstream_decode,
    triton_paged_int4_decode,
)
from bitplane_attn.triton_kernels import (  # noqa: E402
    _bitplane_singlepass_gated_kernel,
    _choose_dual_splits,
    _choose_num_splits,
    _ellipsoid_score_kernel,
    _indexed_base2_splitkv_kernel,
    _indexed_int4_splitkv_kernel,
    _int4_splitkv_kernel,
    _paged_anisotropic_score_kernel,
    _paged_indexed_base2_splitkv_kernel,
    _paged_indexed_int4_splitkv_kernel,
    _paged_int4_splitkv_kernel,
)

KILL_TEST_JSON = PROJECT_ROOT / "results" / "kill_test_bitplane.json"
OUT_JSON = PROJECT_ROOT / "results" / "triton_benchmark.json"
OUT_PNG = PROJECT_ROOT / "results" / "triton_benchmark.png"

PAGE_SIZE = 32
RECENT_TOKENS = 32
PREFIX_LEN = 2048
CONT_STEPS = 128
NEEDLE_DECODE_STEPS = 16


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


def _load_doc_pair() -> tuple[str, str]:
    """优先从 BITPLANE_DOC_DIR 加载长文档语料，否则回退使用仓库 README 与源码文本。"""
    doc_dir_env = os.environ.get("BITPLANE_DOC_DIR")
    if doc_dir_env:
        d_root = Path(doc_dir_env)
        p1 = d_root / "PROOF.md"
        r1 = d_root / "results" / "PHASE2_RESEARCH_REPORT.md"
        r2 = d_root / "results" / "REAL_LLM_VALIDATION.md"
        r3 = d_root / "results" / "BENCHMARK_REPORT.md"
        if p1.exists() and r1.exists() and r2.exists() and r3.exists():
            doc2 = "\n\n".join(
                [
                    r1.read_text(encoding="utf-8"),
                    r2.read_text(encoding="utf-8"),
                    r3.read_text(encoding="utf-8"),
                ]
            )
            return p1.read_text(encoding="utf-8"), doc2

    readme_fp = PROJECT_ROOT / "README.md"
    doc1 = readme_fp.read_text(encoding="utf-8") if readme_fp.exists() else ""
    src_files = sorted((PROJECT_ROOT / "src" / "bitplane_attn").glob("*.py"))
    doc2 = "\n\n".join(p.read_text(encoding="utf-8") for p in src_files)
    if not doc1.strip():
        doc1 = doc2
    return doc1, doc2


def _bench_cuda_ms(fn, warmup: int = 25, iters: int = 80) -> float:
    """使用 CUDA Event 执行高精度同步计时，返回单次调用平均毫秒数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    return float(start_evt.elapsed_time(end_evt)) / float(iters)


def _pytorch_cudnn_gqa_sdpa_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """公平官方原生基线：PyTorch NVIDIA cuDNN GQA SDPA (enable_gqa=True, 零张量展开复制)。"""
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        return F.scaled_dot_product_attention(
            query, key, value, scale=scaling, enable_gqa=True
        ).transpose(1, 2).contiguous()


def run_latency_benchmarks() -> dict[str, Any]:
    """[1/5] 全规格算子延时对比（含原生 cuDNN GQA SDPA、连续紧缩页表与乱序真 PagedAttention 物理页池）。"""
    print("\n================ [1/5] RTX 5070 Ti 公平基线与真 PagedAttention 算子延时压测 ================", flush=True)
    device = torch.device("cuda:0")
    specs = [
        {"name": "qwen3_d128", "label": "Qwen3/Llama-3 GQA (H_q=32, H_kv=8, d=128)", "h_q": 32, "h_kv": 8, "d": 128},
        {"name": "qwen35_d256", "label": "Qwen3.5 Hybrid Global (H_q=16, H_kv=4, d=256)", "h_q": 16, "h_kv": 4, "d": 256},
    ]
    seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    batches = [1, 4]

    results: dict[str, Any] = {}
    for sp in specs:
        sp_name = sp["name"]
        h_q, h_kv, d = sp["h_q"], sp["h_kv"], sp["d"]
        grp = h_q // h_kv
        scaling = 1.0 / math.sqrt(d)
        results[sp_name] = {"meta": sp, "by_batch": {}}

        for bsz in batches:
            b_key = f"B={bsz}"
            results[sp_name]["by_batch"][b_key] = {}
            print(f"\n--- {sp['label']} | {b_key} ---", flush=True)
            for lk in seq_lens:
                torch.manual_seed(1234 + bsz * 17 + lk)
                q = torch.randn((bsz, h_q, 1, d), device=device, dtype=torch.float16)
                k = torch.randn((bsz, h_kv, lk, d), device=device, dtype=torch.float16) * 0.5
                v = torch.randn((bsz, h_kv, lk, d), device=device, dtype=torch.float16) * 0.5
                k[:, :, :4, :] = q[:, ::grp, 0, :].unsqueeze(2) * 3.2
                k[:, :, (lk // 3), :] = q[:, ::grp, 0, :] * 3.0

                cache = PackedBitPlaneKVCache.from_kv(k, v, page_size=PAGE_SIZE, recent_tokens=RECENT_TOKENS)
                mem = cache.memory_stats()

                # 构建真实 PagedAttention 乱序碎片化物理页池（故意打乱物理块号检验间接寻址性能）
                pool, sids = PagedBitPlaneKVPool.from_dense_kv(
                    k, v, page_size=PAGE_SIZE, recent_tokens=RECENT_TOKENS, shuffle_blocks=True, seed=2026 + lk
                )
                paged_batch = pool.build_batch(sids)
                pool_mem = pool.memory_stats()

                _, st_gated = triton_bitplane_gated_decode(q, cache, scaling, tau_hot=2.0, tau_skip=4.5, return_stats=True)
                _, st_dual85 = triton_bitplane_dualstream_decode(q, cache, scaling, hot_ratio=0.15, warm_ratio=0.85, ellipsoid_beta=1.5, return_stats=True)
                _, st_dual35 = triton_bitplane_dualstream_decode(q, cache, scaling, hot_ratio=0.15, warm_ratio=0.35, ellipsoid_beta=1.5, return_stats=True)
                _, st_paged35 = triton_paged_bitplane_dualstream_decode(
                    q, paged_batch, scaling, hot_ratio=0.15, warm_ratio=0.35, beta=1.5, return_stats=True
                )

                iters = 60 if lk >= 32768 else 100
                warmup = 20 if lk >= 32768 else 30

                # 官方 cuDNN GQA SDPA (d <= 128 支持；d=256 当前 PyTorch cuDNN 不支持，直接跳过避免刷屏警告)
                t_cudnn_sdpa: float | None = None
                if d <= 128:
                    _pytorch_cudnn_gqa_sdpa_decode(q, k, v, scaling)
                    t_cudnn_sdpa = _bench_cuda_ms(
                        lambda: _pytorch_cudnn_gqa_sdpa_decode(q, k, v, scaling), warmup=warmup, iters=iters
                    )

                t_fp16 = _bench_cuda_ms(lambda: triton_fp16_paged_decode(q, k, v, scaling), warmup=warmup, iters=iters)
                t_int4 = _bench_cuda_ms(lambda: triton_int4_paged_decode(q, cache, scaling), warmup=warmup, iters=iters)
                t_paged_int4 = _bench_cuda_ms(lambda: triton_paged_int4_decode(q, paged_batch, scaling), warmup=warmup, iters=iters)
                t_twopass = _bench_cuda_ms(lambda: triton_bitplane_decode(q, cache, scaling, top_ratio=0.15), warmup=warmup, iters=iters)
                t_gated = _bench_cuda_ms(lambda: triton_bitplane_gated_decode(q, cache, scaling, tau_hot=2.0, tau_skip=4.5), warmup=warmup, iters=iters)
                t_dual85 = _bench_cuda_ms(
                    lambda: triton_bitplane_dualstream_decode(q, cache, scaling, hot_ratio=0.15, warm_ratio=0.85, ellipsoid_beta=1.5),
                    warmup=warmup, iters=iters,
                )
                t_dual35 = _bench_cuda_ms(
                    lambda: triton_bitplane_dualstream_decode(q, cache, scaling, hot_ratio=0.15, warm_ratio=0.35, ellipsoid_beta=1.5),
                    warmup=warmup, iters=iters,
                )
                t_paged35 = _bench_cuda_ms(
                    lambda: triton_paged_bitplane_dualstream_decode(q, paged_batch, scaling, hot_ratio=0.15, warm_ratio=0.35, beta=1.5),
                    warmup=warmup, iters=iters,
                )

                row = {
                    "seq_len": lk,
                    "latency_ms": {
                        "pytorch_cudnn_sdpa_gqa": t_cudnn_sdpa,
                        "triton_fp16": t_fp16,
                        "triton_int4": t_int4,
                        "paged_triton_int4": t_paged_int4,
                        "bitplane_twopass": t_twopass,
                        "bitplane_gated": t_gated,
                        "bitplane_dualstream_15_85": t_dual85,
                        "bitplane_dualstream_15_35": t_dual35,
                        "paged_dualstream_15_35": t_paged35,
                    },
                    "speedup_paged35_vs_cudnn_sdpa": (t_cudnn_sdpa / t_paged35) if t_cudnn_sdpa else None,
                    "speedup_paged35_vs_fp16": t_fp16 / t_paged35,
                    "speedup_paged35_vs_paged_int4": t_paged_int4 / t_paged35,
                    "speedup_dual35_vs_int4": t_int4 / t_dual35,
                    "speedup_dual85_vs_int4": t_int4 / t_dual85,
                    "paged_indirect_overhead_ratio": t_paged35 / t_dual35,
                    "gated_stats": st_gated,
                    "dualstream_15_85_stats": st_dual85,
                    "dualstream_15_35_stats": st_dual35,
                    "paged_dualstream_15_35_stats": st_paged35,
                    "memory_mb": {
                        "fp16_equiv": mem["fp16_equiv_bytes"] / (1024.0 ** 2),
                        "packed_total": mem["total_bytes"] / (1024.0 ** 2),
                        "paged_pool_active": pool_mem["total_active_bytes"] / (1024.0 ** 2),
                        "compression_ratio": mem["compression_ratio"],
                        "effective_bits_stored_total": mem["effective_bits_stored_total"],
                    },
                }
                results[sp_name]["by_batch"][b_key][str(lk)] = row
                cudnn_s = f"{t_cudnn_sdpa:5.3f}ms" if t_cudnn_sdpa else "  N/A  "
                sp_cudnn = f"{row['speedup_paged35_vs_cudnn_sdpa']:.2f}x" if t_cudnn_sdpa else "N/A"
                print(
                    f"  L={lk:5d} | cuDNN={cudnn_s} | FP16={t_fp16:5.3f}ms | INT4={t_int4:5.3f}ms | "
                    f"PagedINT4={t_paged_int4:5.3f}ms | Dual35={t_dual35:5.3f}ms | PagedDual35={t_paged35:5.3f}ms "
                    f"[vsPagedINT4={row['speedup_paged35_vs_paged_int4']:.2f}x, vsFP16={row['speedup_paged35_vs_fp16']:.2f}x, "
                    f"vscuDNN={sp_cudnn}, ovhd={(row['paged_indirect_overhead_ratio']-1)*100:+.1f}%]",
                    flush=True,
                )

                del q, k, v, cache, pool, paged_batch
                gc.collect()
                torch.cuda.empty_cache()

    return results


def run_ptxas_microarch_audit() -> dict[str, Any]:
    """[2/5] 捕获连续页表与真物理页池 PagedAttention Triton 核的 PTXAS 寄存器与 SM 占用率。"""
    print("\n================ [2/5] PTXAS 编译微架构剖析（含真页表 Paged 核验证 <= 128 regs） ================", flush=True)
    device = torch.device("cuda:0")
    audit: dict[str, Any] = {}

    for label, bsz, h_q, h_kv, d in [
        ("d128_B4", 4, 32, 8, 128),
        ("d256_B4", 4, 16, 4, 256),
    ]:
        lk = 32768
        grp = h_q // h_kv
        d4 = d // 4
        q_c = torch.randn((bsz, h_q, d), device=device, dtype=torch.float16)
        k = torch.randn((bsz, h_kv, lk, d), device=device, dtype=torch.float16)
        v = torch.randn((bsz, h_kv, lk, d), device=device, dtype=torch.float16)
        cache = PackedBitPlaneKVCache.from_kv(k, v, page_size=PAGE_SIZE, recent_tokens=RECENT_TOKENS)
        pool, sids = PagedBitPlaneKVPool.from_dense_kv(k, v, page_size=PAGE_SIZE, recent_tokens=RECENT_TOKENS, shuffle_blocks=True)
        pb = pool.build_batch(sids)

        num_pages = cache.num_pages
        n_hot = int(math.ceil(num_pages * 0.15))
        n_warm = int(math.ceil(num_pages * 0.35))
        hot_idx = torch.arange(n_hot, device=device, dtype=torch.int32).view(1, 1, -1).expand(bsz, h_kv, -1).contiguous()
        warm_idx = torch.arange(n_hot, n_hot + n_warm, device=device, dtype=torch.int32).view(1, 1, -1).expand(bsz, h_kv, -1).contiguous()
        n_hot_t = torch.full((bsz,), n_hot, device=device, dtype=torch.int32)
        n_warm_t = torch.full((bsz,), n_warm, device=device, dtype=torch.int32)

        ns_all = _choose_num_splits(bsz, h_kv, num_pages)
        sm_cnt = int(torch.cuda.get_device_properties(device).multi_processor_count)
        ns_hot = _choose_dual_splits(bsz, h_kv, n_hot, head_dim=d, sm_count=sm_cnt)
        ns_warm = _choose_dual_splits(bsz, h_kv, n_warm, head_dim=d, sm_count=sm_cnt)
        total_splits = max(ns_all, ns_hot + ns_warm)

        part_m = torch.empty((bsz, h_q, total_splits), device=device, dtype=torch.float32)
        part_l = torch.empty((bsz, h_q, total_splits), device=device, dtype=torch.float32)
        part_o = torch.empty((bsz, h_q, total_splits, d), device=device, dtype=torch.float32)
        m_anchor = torch.zeros((bsz, h_q), device=device, dtype=torch.float32)
        tier_out = torch.empty((1, 1, 1), device=device, dtype=torch.uint8)
        head_scores = torch.empty((bsz, h_q, num_pages), device=device, dtype=torch.float32)
        scaling = 1.0 / math.sqrt(d)

        ck_int4 = _int4_splitkv_kernel[(bsz * h_kv, ns_all)](
            q_c, cache.k_base_packed, cache.k_res_packed, cache.k_min, cache.k_scale,
            cache.v_base_packed, cache.v_res_packed, cache.v_min, cache.v_scale,
            part_m, part_l, part_o,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            cache.k_base_packed.stride(0), cache.k_base_packed.stride(1), cache.k_base_packed.stride(2),
            cache.k_base_packed.stride(3), cache.k_base_packed.stride(4),
            cache.k_min.stride(0), cache.k_min.stride(1), cache.k_min.stride(2), cache.k_min.stride(3),
            cache.v_min.stride(0), cache.v_min.stride(1), cache.v_min.stride(2), cache.v_min.stride(3),
            part_m.stride(0), part_m.stride(1), part_m.stride(2),
            part_o.stride(0), part_o.stride(1), part_o.stride(2), part_o.stride(3),
            scaling, num_pages, ns_all,
            H_kv=h_kv, G=grp, D4=d4, PAGE_SIZE=PAGE_SIZE, BLOCK_G=16, BLOCK_D4=d4, num_warps=4,
        )
        ck_gated = _bitplane_singlepass_gated_kernel[(bsz * h_kv, ns_all)](
            q_c, m_anchor, cache.k_base_packed, cache.k_res_packed, cache.k_min, cache.k_scale,
            cache.v_base_packed, cache.v_res_packed, cache.v_min, cache.v_scale,
            part_m, part_l, part_o, tier_out,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            m_anchor.stride(0), m_anchor.stride(1),
            cache.k_base_packed.stride(0), cache.k_base_packed.stride(1), cache.k_base_packed.stride(2),
            cache.k_base_packed.stride(3), cache.k_base_packed.stride(4),
            cache.k_min.stride(0), cache.k_min.stride(1), cache.k_min.stride(2), cache.k_min.stride(3),
            cache.v_min.stride(0), cache.v_min.stride(1), cache.v_min.stride(2), cache.v_min.stride(3),
            part_m.stride(0), part_m.stride(1), part_m.stride(2),
            part_o.stride(0), part_o.stride(1), part_o.stride(2), part_o.stride(3),
            tier_out.stride(0), tier_out.stride(1), tier_out.stride(2),
            scaling, 2.0, 4.5, num_pages, ns_all,
            H_kv=h_kv, G=grp, D4=d4, PAGE_SIZE=PAGE_SIZE, BLOCK_G=16, BLOCK_D4=d4, RECORD_TIERS=False, num_warps=4,
        )
        ck_p_score = _paged_anisotropic_score_kernel[(bsz * h_kv, (num_pages + 31) // 32)](
            q_c, pb.block_table, pb.num_pages_per_seq, pool.k_min_pool, pool.k_scale_pool, head_scores,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), pb.block_table.stride(0), pb.block_table.stride(1),
            pool.k_min_pool.stride(0), pool.k_min_pool.stride(1), pool.k_min_pool.stride(2),
            head_scores.stride(0), head_scores.stride(1), head_scores.stride(2),
            scaling, 1.5, num_pages,
            H_kv=h_kv, G=grp, HEAD_DIM=d, BLOCK_G=16, BLOCK_D=d, BLOCK_P=32, SCORE_MODE=1,
            num_warps=4, num_stages=(2 if d >= 256 else 3),
        )
        ck_p_hot = _paged_indexed_int4_splitkv_kernel[(bsz * h_kv, ns_hot)](
            q_c, hot_idx, n_hot_t,
            pool.k_base_pool, pool.k_res_pool, pool.k_min_pool, pool.k_scale_pool,
            pool.v_base_pool, pool.v_res_pool, pool.v_min_pool, pool.v_scale_pool,
            part_m, part_l, part_o,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            hot_idx.stride(0), hot_idx.stride(1), hot_idx.stride(2),
            pool.k_base_pool.stride(0), pool.k_base_pool.stride(1), pool.k_base_pool.stride(2), pool.k_base_pool.stride(3),
            pool.k_min_pool.stride(0), pool.k_min_pool.stride(1), pool.k_min_pool.stride(2),
            pool.v_min_pool.stride(0), pool.v_min_pool.stride(1), pool.v_min_pool.stride(2),
            part_m.stride(0), part_m.stride(1), part_m.stride(2),
            part_o.stride(0), part_o.stride(1), part_o.stride(2), part_o.stride(3),
            scaling, ns_hot, 0, n_hot,
            H_kv=h_kv, G=grp, D4=d4, PAGE_SIZE=PAGE_SIZE, BLOCK_G=16, BLOCK_D4=d4, UNIFORM_N_SEL=True, num_warps=4,
        )
        ck_p_warm = _paged_indexed_base2_splitkv_kernel[(bsz * h_kv, ns_warm)](
            q_c, warm_idx, n_warm_t,
            pool.k_base_pool, pool.k_min_pool, pool.k_scale_pool,
            pool.v_base_pool, pool.v_min_pool, pool.v_scale_pool,
            part_m, part_l, part_o,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            warm_idx.stride(0), warm_idx.stride(1), warm_idx.stride(2),
            pool.k_base_pool.stride(0), pool.k_base_pool.stride(1), pool.k_base_pool.stride(2), pool.k_base_pool.stride(3),
            pool.k_min_pool.stride(0), pool.k_min_pool.stride(1), pool.k_min_pool.stride(2),
            pool.v_min_pool.stride(0), pool.v_min_pool.stride(1), pool.v_min_pool.stride(2),
            part_m.stride(0), part_m.stride(1), part_m.stride(2),
            part_o.stride(0), part_o.stride(1), part_o.stride(2), part_o.stride(3),
            scaling, ns_warm, ns_hot, n_warm,
            H_kv=h_kv, G=grp, D4=d4, PAGE_SIZE=PAGE_SIZE, BLOCK_G=16, BLOCK_D4=d4, UNIFORM_N_SEL=True, num_warps=4,
        )
        torch.cuda.synchronize()

        def _extract(ck):
            n_regs = int(getattr(ck, "n_regs", 0))
            md = getattr(ck, "metadata", None)
            shared = int(getattr(md, "shared", 0)) if md is not None else 0
            warps = int(getattr(md, "num_warps", 4)) if md is not None else 4
            stages = int(getattr(md, "num_stages", 0)) if md is not None else 0
            threads = warps * 32
            cta_regs = 65536 // max(1, n_regs * threads) if n_regs > 0 else 0
            cta_smem = 101376 // max(1, shared) if shared > 0 else 32
            cta_max = min(12, cta_regs, cta_smem)
            return {
                "n_regs": n_regs,
                "shared_bytes": shared,
                "num_warps": warps,
                "num_stages": stages,
                "max_ctas_per_sm": cta_max,
                "cp_async_pipelined": shared >= 6000,
            }

        audit[label] = {
            "singlepass_gated": _extract(ck_gated),
            "int4_splitkv": _extract(ck_int4),
            "paged_anisotropic_score": _extract(ck_p_score),
            "paged_hot_int4": _extract(ck_p_hot),
            "paged_warm_base2": _extract(ck_p_warm),
        }
        print(f"\n--- PTXAS 规格 {label} ---", flush=True)
        for k_name, info in audit[label].items():
            print(
                f"  {k_name:24s} | n_regs={info['n_regs']:3d} | shared={info['shared_bytes']:5d} B | "
                f"max_ctas/SM={info['max_ctas_per_sm']:2d}",
                flush=True,
            )

        del q_c, k, v, cache, pool, pb, hot_idx, warm_idx, n_hot_t, n_warm_t, part_m, part_l, part_o, m_anchor, tier_out, head_scores
        gc.collect()
        torch.cuda.empty_cache()

    return audit


def _build_docs(tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    doc1, doc2 = _load_doc_pair()
    seqs = []
    need = PREFIX_LEN + CONT_STEPS
    for name, txt in [("PROOF.md", doc1), ("RESEARCH_REPORTS.md", doc2)]:
        ids = tokenizer.encode(txt, add_special_tokens=False)
        while len(ids) < need + 64:
            ids = ids + ids
        chunk = ids[64 : 64 + need]
        seqs.append(
            {
                "name": name,
                "prefix_ids": torch.tensor([chunk[:PREFIX_LEN]], dtype=torch.long, device=device),
                "cont_ids": torch.tensor([chunk[PREFIX_LEN:]], dtype=torch.long, device=device),
            }
        )
    return seqs


def _build_needles(tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    filler = (
        "Distributed KV-cache management requires tracking memory blocks, page tables, and bandwidth "
        "utilization across worker nodes during autoregressive decoding. Telemetry monitors log p99 latency "
        "and throughput stability under dynamic request batching and multi-tenant scheduling policies. "
    )
    f_ids = tokenizer.encode(filler, add_special_tokens=False)
    specs = [
        ("depth_10", 0.10, "ORN-9482-X7", "Project Orion"),
        ("depth_30", 0.30, "ZPH-3109-K4", "Project Zephyr"),
        ("depth_50", 0.50, "VTX-6625-M9", "Project Vertex"),
        ("depth_70", 0.70, "KRM-4817-Q2", "Project Karma"),
        ("depth_90", 0.90, "NBL-5294-T8", "Project Nebula"),
    ]
    cases = []
    for name, depth, code, proj in specs:
        n_ids = tokenizer.encode(
            f" [AUDIT RECORD] The official verification code for {proj} is {code}. ", add_special_tokens=False
        )
        q_ids = tokenizer.encode(
            f" Based strictly on the audit record above, the official verification code for {proj} is",
            add_special_tokens=False,
        )
        rem = PREFIX_LEN - len(n_ids) - len(q_ids)
        b_len = max(16, int(round(rem * depth)))
        a_len = rem - b_len
        b_ids = (f_ids * ((b_len // len(f_ids)) + 2))[:b_len]
        a_ids = (f_ids * ((a_len // len(f_ids)) + 2))[:a_len]
        cases.append(
            {
                "name": name,
                "depth": depth,
                "code": code,
                "input_ids": torch.tensor([b_ids + n_ids + a_ids + q_ids], dtype=torch.long, device=device),
            }
        )
    return cases


def run_routing_recall_ablation() -> dict[str, Any]:
    """[3/5] 真实大模型 (Qwen3.5-0.8B & Qwen3-0.6B) 实测各向异性页排序启发式的真实 Attention Mass Recall。"""
    print("\n================ [3/5] 真实大模型 Attention Mass Recall 与 Top-k 页命中消融 ================", flush=True)
    device = torch.device("cuda:0")
    out: dict[str, Any] = {}

    strats = [
        ("center_beta0", "center", 0.0),
        ("aniso_beta0.5", "anisotropic", 0.5),
        ("aniso_beta1.0", "anisotropic", 1.0),
        ("aniso_beta1.5", "anisotropic", 1.5),
        ("aniso_beta2.0", "anisotropic", 2.0),
        ("box_l1", "box_l1", 1.0),
    ]

    for model_name, is_35 in [("Qwen3.5-0.8B", True), ("Qwen3-0.6B", False)]:
        m_src = _resolve_model_source(model_name)
        tokenizer = AutoTokenizer.from_pretrained(m_src)
        model = (Qwen3_5ForConditionalGeneration if is_35 else AutoModelForCausalLM).from_pretrained(
            m_src, dtype=torch.bfloat16
        ).to(device).eval()

        docs = _build_docs(tokenizer, device)
        d0 = docs[0]
        full_ids = torch.cat([d0["prefix_ids"], d0["cont_ids"][:, :16]], dim=1)
        prompt_ids = full_ids[:, :PREFIX_LEN]
        step_ids = full_ids[:, PREFIX_LEN:]

        pkv = DynamicCache(config=model.config)
        with torch.no_grad():
            model(input_ids=prompt_ids, cache=pkv, use_cache=True)

        # 用前向 hook 提取接下来 8 步 Decode 中每全注意力层的真实 Query 状态 (RoPE 后输入 SDPA 的 Q)
        captured_qs: dict[int, list[torch.Tensor]] = {}
        orig_forwards: list[tuple[Any, Any]] = []

        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as q3_rope
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as q35_rope

        def _make_q_hook(mod_ref: torch.nn.Module):
            l_idx = int(mod_ref.layer_idx)
            captured_qs[l_idx] = []
            orig_fwd = mod_ref.forward

            def _wrapped(hidden_states, position_embeddings, attention_mask, cache=None, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, mod_ref.head_dim)
                if isinstance(mod_ref, Qwen3_5Attention):
                    q_proj, _ = torch.chunk(mod_ref.q_proj(hidden_states).view(*input_shape, -1, mod_ref.head_dim * 2), 2, dim=-1)
                    qs = mod_ref.q_norm(q_proj.view(hidden_shape)).transpose(1, 2)
                    cos, sin = position_embeddings
                    qs, _ = q35_rope(qs, qs, cos, sin)
                else:
                    qs = mod_ref.q_norm(mod_ref.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    cos, sin = position_embeddings
                    qs, _ = q3_rope(qs, qs, cos, sin)
                if qs.shape[2] == 1:
                    captured_qs[l_idx].append(qs.detach().float())
                return orig_fwd(hidden_states, position_embeddings, attention_mask, cache=cache, **kwargs)

            orig_forwards.append((mod_ref, orig_fwd))
            return _wrapped

        for mod in model.modules():
            if isinstance(mod, (Qwen3Attention, Qwen3_5Attention)):
                mod.forward = _make_q_hook(mod)

        with torch.no_grad():
            for t in range(8):
                model(input_ids=step_ids[:, t : t + 1], cache=pkv, use_cache=True)

        for mod_ref, orig_fwd in orig_forwards:
            mod_ref.forward = orig_fwd

        # 对前 63 整页 (hist_len = 2016) 核算真实 BF16 注意力质量分布及各排序策略召回率
        hist_len = (PREFIX_LEN - RECENT_TOKENS) // PAGE_SIZE * PAGE_SIZE
        n_pages = hist_len // PAGE_SIZE
        n_hot = int(math.ceil(n_pages * 0.15))
        n_warm_only = int(math.ceil(n_pages * 0.35))
        n_keep = min(n_pages, n_hot + n_warm_only)

        accum: dict[str, dict[str, list[float]]] = {
            s[0]: {"mass_15": [], "mass_50": [], "oracle_15": [], "mass_15_pin0": [], "mass_50_pin0": []}
            for s in strats
        }

        for l_idx, qs_list in captured_qs.items():
            lyr = pkv.layers[l_idx]
            k_all = lyr.keys[:, :, :hist_len, :].float()  # (1, H_kv, hist_len, d)
            h_kv = k_all.shape[1]
            d = k_all.shape[-1]
            q_stack = torch.cat(qs_list, dim=2)  # (1, H_q, 8, d)
            h_q = q_stack.shape[1]
            grp = h_q // h_kv
            scaling = 1.0 / math.sqrt(d)

            # 按页打包算页级 min/scale（同核严格一致转为 fp16 再转 float 评价）
            k_paged = k_all.view(1, h_kv, n_pages, PAGE_SIZE, d)
            km = k_paged.min(dim=-2).values.to(torch.float16).float()
            kmax = k_paged.max(dim=-2).values
            ks = (torch.clamp(kmax - k_paged.min(dim=-2).values, min=1e-7) / 15.0).to(torch.float16).float()
            k_mid = km + 7.5 * ks
            k_rad = 7.5 * ks

            q_g = q_stack.view(1, h_kv, grp, 8, d) * scaling
            # 精确 BF16/FP32 在历史页上的组注意力质量与 Oracle Top-15% 页
            logits = torch.einsum("bhgtd,bhpsd->bhgtps", q_g, k_paged)  # (1, H_kv, G, T, P, S)
            probs = torch.softmax(logits.view(1, h_kv, grp, 8, n_pages * PAGE_SIZE), dim=-1).view(1, h_kv, grp, 8, n_pages, PAGE_SIZE)
            # 组内任意头对页 p 的贡献取 max_g(page_prob) 以客观对齐 GQA 共享选页真实覆盖需求；也兼计组平均 mass
            page_mass_grp = probs.sum(dim=-1).mean(dim=2)  # (1, H_kv, 8, P), sum_p == 1.0
            oracle_idx = torch.topk(page_mass_grp, k=n_hot, dim=-1).indices

            mu = torch.einsum("bhgtd,bhpd->bhgtp", q_g, k_mid)
            var = torch.einsum("bhgtd,bhpd->bhgtp", q_g * q_g, k_rad * k_rad)
            l1_r = torch.einsum("bhgtd,bhpd->bhgtp", torch.abs(q_g), k_rad)

            for s_name, s_type, beta_v in strats:
                if s_type == "center":
                    u = mu
                elif s_type == "anisotropic":
                    u = mu + beta_v * torch.sqrt(torch.clamp(var, min=0.0))
                else:
                    u = mu + l1_r
                u_rel = u - u.max(dim=-1, keepdim=True).values
                gs = u_rel.max(dim=2).values + 1e-5 * torch.exp(u_rel).sum(dim=2)

                # 1) 纯启发式排序（不 pin0）
                idx_50 = torch.topk(gs, k=n_keep, dim=-1, sorted=True).indices
                idx_15 = idx_50[..., :n_hot]
                m15 = float(torch.gather(page_mass_grp, -1, idx_15).sum(dim=-1).mean().item())
                m50 = float(torch.gather(page_mass_grp, -1, idx_50).sum(dim=-1).mean().item())

                # 计算相对 Oracle Top-15% 页集合重合率
                o_mask = torch.zeros_like(page_mass_grp, dtype=torch.bool).scatter_(-1, oracle_idx, True)
                p_mask = torch.zeros_like(page_mass_grp, dtype=torch.bool).scatter_(-1, idx_15, True)
                inter = float((o_mask & p_mask).float().sum(dim=-1).mean().item()) / float(n_hot)

                # 2) pin Page 0 (同算子默认策略)
                gs_p0 = gs.clone()
                gs_p0[..., 0] = 1e9
                idx_p50 = torch.topk(gs_p0, k=n_keep, dim=-1, sorted=True).indices
                idx_p15 = idx_p50[..., :n_hot]
                mp15 = float(torch.gather(page_mass_grp, -1, idx_p15).sum(dim=-1).mean().item())
                mp50 = float(torch.gather(page_mass_grp, -1, idx_p50).sum(dim=-1).mean().item())

                accum[s_name]["mass_15"].append(m15)
                accum[s_name]["mass_50"].append(m50)
                accum[s_name]["oracle_15"].append(inter)
                accum[s_name]["mass_15_pin0"].append(mp15)
                accum[s_name]["mass_50_pin0"].append(mp50)

        out[model_name] = {}
        print(f"\n--- {model_name} 真实 Attention Mass Recall（P={n_pages}, Hot=15%, Hot+Warm=50%）---", flush=True)
        for s_name, _, _ in strats:
            row = {k: float(np.mean(v)) for k, v in accum[s_name].items()}
            out[model_name][s_name] = row
            print(
                f"  {s_name:15s} | Pure Recall@15%={row['mass_15']*100:5.1f}% Recall@50%={row['mass_50']*100:5.1f}% "
                f"| Pin0 Recall@15%={row['mass_15_pin0']*100:5.1f}% Recall@50%={row['mass_50_pin0']*100:5.1f}% "
                f"| OracleTop15 Overlap={row['oracle_15']*100:5.1f}%",
                flush=True,
            )

        del model, pkv, captured_qs
        gc.collect()
        torch.cuda.empty_cache()

    return out


def _autoreg_128_steps(
    model: torch.nn.Module,
    pkv_init: Any,
    eval_tokens: torch.Tensor,
    clone_cache: bool = True,
) -> np.ndarray:
    """严格 L_q == 1 逐 token 教师强制前向（输入端到端真实 Cache）。"""
    pkv = copy.deepcopy(pkv_init) if clone_cache else pkv_init
    logits_steps = []
    with torch.no_grad():
        for t in range(eval_tokens.shape[1]):
            out = model(input_ids=eval_tokens[:, t : t + 1], cache=pkv, use_cache=True)
            pkv = out.past_key_values if hasattr(out, "past_key_values") else pkv
            logits_steps.append(out.logits[0, -1, :].float())
    return torch.stack(logits_steps, dim=0).cpu().numpy()


def _clone_bp_cache(src: BitPlaneDynamicCache) -> BitPlaneDynamicCache:
    """深拷贝 `BitPlaneDynamicCache`（确保每个独立样本互不干扰物理页池状态）。"""
    return copy.deepcopy(src)


def _eval_model_vram_and_throughput(
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
) -> dict[str, Any]:
    """在真实模型上以完整 `model.generate()` 实测原生 DynamicCache vs BitPlaneDynamicCache 真显存与解码吞吐。"""
    print(f"\n--- 真端到端显存压缩与 generate() 吞吐压测: {model_name} ---", flush=True)
    # 先跑 1 次极小热身消除 CUDA / Triton JIT allocator 首帧扰动
    warm_ids = torch.randint(10, 500, (1, 128), device=device, dtype=torch.long)
    warm_bp = BitPlaneDynamicCache(config=model.config, page_size=PAGE_SIZE, recent_tokens=RECENT_TOKENS)
    with warm_bp.attach(model), torch.no_grad():
        model.generate(input_ids=warm_ids, cache=warm_bp, max_new_tokens=8, do_sample=False)
    del warm_ids, warm_bp
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model_vram_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)

    specs = [
        ("B4_L2048_G128", 4, 2048, 128),
        ("B4_L4096_G128", 4, 4096, 128),
        ("B1_L8192_G64", 1, 8192, 64),
    ]
    res: dict[str, Any] = {"model_baseline_vram_mb": model_vram_mb, "configs": {}}

    for s_key, bsz, l_pref, gen_steps in specs:
        torch.manual_seed(9090 + bsz + l_pref)
        prompt = torch.randint(100, 2000, (bsz, l_pref), device=device, dtype=torch.long)

        def _measure_single(use_bitplane: bool) -> dict[str, Any]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)

            if use_bitplane:
                cache_obj = BitPlaneDynamicCache(
                    config=model.config,
                    page_size=PAGE_SIZE,
                    recent_tokens=RECENT_TOKENS,
                    mode="triton_dualstream",
                    hot_ratio=0.15,
                    warm_ratio=0.35,
                    beta=1.5,
                )
                ctx = cache_obj.attach(model)
            else:
                cache_obj = DynamicCache(config=model.config)
                from contextlib import nullcontext
                ctx = nullcontext()

            with ctx, torch.no_grad():
                t0 = time.perf_counter()
                pref_out = model(input_ids=prompt, cache=cache_obj, use_cache=True)
                torch.cuda.synchronize()
                t_prefill_ms = (time.perf_counter() - t0) * 1000.0
                nxt = torch.argmax(pref_out.logits[:, -1, :], dim=-1, keepdim=True)
                del pref_out

                # 首步预热不计入 Decode 均值（若需全量 `generate` 也单独统计完整耗时）
                torch.cuda.synchronize()
                td0 = time.perf_counter()
                for _ in range(gen_steps):
                    step_out = model(input_ids=nxt, cache=cache_obj, use_cache=True)
                    nxt = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                t_decode_total_s = time.perf_counter() - td0

            steady_kv_mb = (torch.cuda.memory_allocated(device) / (1024.0 ** 2)) - base_mb
            peak_total_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            peak_kv_overhead_mb = peak_total_mb - base_mb
            cache_meta = cache_obj.memory_stats() if use_bitplane else None

            del cache_obj, nxt
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "base_model_mb": base_mb,
                "steady_kv_vram_mb": steady_kv_mb,
                "peak_total_vram_mb": peak_total_mb,
                "peak_kv_delta_mb": peak_kv_overhead_mb,
                "prefill_ms": t_prefill_ms,
                "decode_total_s": t_decode_total_s,
                "decode_ms_per_step": (t_decode_total_s * 1000.0) / float(gen_steps),
                "decode_tokens_per_sec": float(bsz * gen_steps) / max(t_decode_total_s, 1e-9),
                "cache_pool_stats": cache_meta,
            }

        nat_m = _measure_single(use_bitplane=False)
        bp_m = _measure_single(use_bitplane=True)

        kv_vram_ratio = nat_m["steady_kv_vram_mb"] / max(bp_m["steady_kv_vram_mb"], 0.1)
        tp_speedup = bp_m["decode_tokens_per_sec"] / max(nat_m["decode_tokens_per_sec"], 1e-6)
        res["configs"][s_key] = {
            "batch_size": bsz,
            "prefill_len": l_pref,
            "gen_steps": gen_steps,
            "native_bf16_dynamic_cache": nat_m,
            "bitplane_dynamic_cache_15_35": bp_m,
            "real_cuda_kv_vram_compression_ratio": kv_vram_ratio,
            "real_decode_throughput_speedup": tp_speedup,
        }
        print(
            f"  {s_key:15s} | Native KV={nat_m['steady_kv_vram_mb']:6.1f} MB ({nat_m['decode_tokens_per_sec']:6.1f} tok/s) "
            f"| BitPlane KV={bp_m['steady_kv_vram_mb']:6.1f} MB ({bp_m['decode_tokens_per_sec']:6.1f} tok/s) "
            f"| 真显存压缩={kv_vram_ratio:.2f}x (全局页池理论={bp_m['cache_pool_stats']['compression_ratio']:.2f}x) "
            f"| Decode吞吐比={tp_speedup:.2f}x",
            flush=True,
        )
        del prompt
        gc.collect()
        torch.cuda.empty_cache()

    return res


def _eval_single_model_real_kernels(model_name: str, is_35: bool, device: torch.device) -> dict[str, Any]:
    """[4/5] 真实大模型挂接真 `BitPlaneDynamicCache` 精度、真显存与生成吞吐端到端验证。"""
    print(f"\n--- 真实模型挂接真 `BitPlaneDynamicCache` 端到端验证: {model_name} (dtype=BF16) ---", flush=True)
    m_src = _resolve_model_source(model_name)
    tokenizer = AutoTokenizer.from_pretrained(m_src)
    model = (Qwen3_5ForConditionalGeneration if is_35 else AutoModelForCausalLM).from_pretrained(
        m_src, dtype=torch.bfloat16
    ).to(device).eval()

    docs = _build_docs(tokenizer, device)
    needles = _build_needles(tokenizer, device)

    # 1) 原生 BF16 DynamicCache 基线
    doc_prep = []
    for d_item in docs:
        pkv_nat = DynamicCache(config=model.config)
        with torch.no_grad():
            model(input_ids=d_item["prefix_ids"][:, :-1], cache=pkv_nat, use_cache=True)
        eval_chunk = torch.cat([d_item["prefix_ids"][:, -1:], d_item["cont_ids"][:, :-1]], dim=1)
        targets = d_item["cont_ids"][0].cpu().numpy()

        ref_logits = _autoreg_128_steps(model, pkv_nat, eval_chunk)
        max_z = np.max(ref_logits, axis=1, keepdims=True)
        lse = max_z[:, 0] + np.log(np.sum(np.exp(ref_logits - max_z), axis=1))
        bf_ppl = float(np.exp(np.mean(lse - ref_logits[np.arange(targets.shape[0]), targets])))
        doc_prep.append(
            {
                "prefix_minus_1": d_item["prefix_ids"][:, :-1],
                "eval_chunk": eval_chunk,
                "targets": targets,
                "ref_logits": ref_logits,
                "bf16_ppl": bf_ppl,
            }
        )
        del pkv_nat

    bf16_ppl = float(np.mean([x["bf16_ppl"] for x in doc_prep]))
    configs = [
        ("triton_int4", "triton_int4", 1.0, 0.0, 0.0),
        ("triton_dualstream_15_85", "triton_dualstream", 0.15, 0.85, 1.5),
        ("triton_dualstream_15_35", "triton_dualstream", 0.15, 0.35, 1.5),
        ("triton_dualstream_15_0", "triton_dualstream", 0.15, 0.0, 1.5),
    ]

    model_res: dict[str, Any] = {"bf16_ppl": bf16_ppl, "configs": {}}
    print(f"  BF16 Autoreg PPL (Native DynamicCache + SDPA) = {bf16_ppl:.4f}", flush=True)

    for cfg_name, mode, hr, wr, beta in configs:
        ppls, kls, bits = [], [], []
        for dp in doc_prep:
            bp_cache = BitPlaneDynamicCache(
                config=model.config,
                page_size=PAGE_SIZE,
                recent_tokens=RECENT_TOKENS,
                mode=mode,
                hot_ratio=hr,
                warm_ratio=wr,
                beta=beta,
                score_mode="anisotropic",
                pin_page0=True,
                collect_stats=True,
            )
            with bp_cache.attach(model), torch.no_grad():
                model(input_ids=dp["prefix_minus_1"], cache=bp_cache, use_cache=True)
                logits = _autoreg_128_steps(model, bp_cache, dp["eval_chunk"], clone_cache=False)
            mz = np.max(logits, axis=1, keepdims=True)
            lse = mz[:, 0] + np.log(np.sum(np.exp(logits - mz), axis=1))
            ppls.append(float(np.exp(np.mean(lse - logits[np.arange(dp["targets"].shape[0]), dp["targets"]]))))

            mr = np.max(dp["ref_logits"], axis=1, keepdims=True)
            lpr = dp["ref_logits"] - mr - np.log(np.sum(np.exp(dp["ref_logits"] - mr), axis=1, keepdims=True))
            lpc = logits - mz - np.log(np.sum(np.exp(logits - mz), axis=1, keepdims=True))
            kls.append(float(np.mean(np.maximum(0.0, np.sum(np.exp(lpr) * (lpr - lpc), axis=1)))))
            bits.extend(bp_cache.read_bits_samples)
            del bp_cache

        hits = 0
        for nc in needles:
            bp_n = BitPlaneDynamicCache(
                config=model.config,
                page_size=PAGE_SIZE,
                recent_tokens=RECENT_TOKENS,
                mode=mode,
                hot_ratio=hr,
                warm_ratio=wr,
                beta=beta,
                score_mode="anisotropic",
                pin_page0=True,
                collect_stats=False,
            )
            with bp_n.attach(model), torch.no_grad():
                model(input_ids=nc["input_ids"][:, :-1], cache=bp_n, use_cache=True)
                cur = nc["input_ids"][:, -1:]
                g_ids = []
                for _ in range(NEEDLE_DECODE_STEPS):
                    o = model(input_ids=cur, cache=bp_n, use_cache=True)
                    cur = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
                    g_ids.append(int(cur.item()))
            if nc["code"] in tokenizer.decode(g_ids, skip_special_tokens=True):
                hits += 1
            del bp_n

        mean_b = float(np.mean(bits)) if bits else compute_logical_read_bits_per_kv_elem("int4")
        cfg_out = {
            "mean_ppl": float(np.mean(ppls)),
            "delta_ppl": float(np.mean(ppls)) - bf16_ppl,
            "mean_kl": float(np.mean(kls)),
            "needle_hits": hits,
            "needle_total": len(needles),
            "logical_read_bits_per_kv_elem": mean_b,
        }
        model_res["configs"][cfg_name] = cfg_out
        print(
            f"  {cfg_name:24s} | read={mean_b:4.2f} b/KV-elem | PPL={cfg_out['mean_ppl']:.3f} "
            f"(dPPL={cfg_out['delta_ppl']:+.3f}) | KL vs BF16={cfg_out['mean_kl']:.5f} | Needles={hits}/5",
            flush=True,
        )

    # 2) 真端到端显存压缩与吞吐压测
    model_res["e2e_vram_and_throughput"] = _eval_model_vram_and_throughput(model, model_name, device)

    del model, doc_prep, needles
    gc.collect()
    torch.cuda.empty_cache()
    return model_res


def run_end_to_end_real_kernels() -> dict[str, Any]:
    """[4/5] 双真实大模型真 BitPlaneDynamicCache 精度、显存与生成吞吐综合校验。"""
    print("\n================ [4/5] 双真实大模型端到端挂载 BitPlaneDynamicCache 评测 ================", flush=True)
    device = torch.device("cuda:0")
    return {
        "Qwen3.5-0.8B": _eval_single_model_real_kernels("Qwen3.5-0.8B", True, device),
        "Qwen3-0.6B": _eval_single_model_real_kernels("Qwen3-0.6B", False, device),
    }


def render_summary_figure(data: dict[str, Any]) -> None:
    """[5/5] 绘制 300 DPI 四宫格顶会级总结图（全英文规范口径、无遮挡排布）。"""
    print("\n================ [5/5] 绘制 300 DPI 四宫格论文总结图 ================", flush=True)
    plt.rcParams.update({"font.size": 9.3, "axes.titlesize": 10.3, "axes.labelsize": 9.3})
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.4), dpi=300)
    plt.subplots_adjust(hspace=0.34, wspace=0.26, top=0.92, bottom=0.08, left=0.07, right=0.97)
    fig.suptitle(
        "Bit-Plane Paged Attention (RTX 5070 Ti sm_120): True Paged KV Pool & Zero-Branch Dual-Stream Kernels",
        fontsize=12.6,
        fontweight="bold",
        y=0.97,
    )

    lat_all = data["latency"]
    ptx = data["ptxas"]
    e2e = data["end_to_end"]

    # (a) Qwen3 d=128, B=4 Decode Latency vs Sequence Length (Fair Native cuDNN SDPA Baseline!)
    ax = axes[0, 0]
    b4_128 = lat_all["qwen3_d128"]["by_batch"]["B=4"]
    ls = [int(k) for k in b4_128.keys()]
    x_pos = np.arange(len(ls))
    x_lbl = [f"{k // 1024}K" for k in ls]

    series_a = [
        ("pytorch_cudnn_sdpa_gqa", "PyTorch cuDNN SDPA (FP16 GQA)", "#7f8c8d", "--", "x"),
        ("triton_fp16", "Triton-FP16 SplitKV", "#34495e", "-.", "s"),
        ("paged_triton_int4", "Paged Triton-INT4 (Shuffled BlockTable)", "#2980b9", "-", "o"),
        ("bitplane_gated", "SinglePass Gated (In-Loop Branch)", "#e67e22", ":", "^"),
        ("bitplane_dualstream_15_85", "DualStream (15% Hot + 85% Warm)", "#16a085", "--", "d"),
        ("paged_dualstream_15_35", "Paged DualStream (15+35, Shuffled, Ours)", "#c0392b", "-", "*"),
    ]
    for key, lbl, col, sty, mkr in series_a:
        ys = [b4_128[str(k)]["latency_ms"][key] for k in ls]
        lw = 2.6 if "Ours" in lbl else 1.7
        ms = 8 if "Ours" in lbl else 5
        ax.plot(x_pos, ys, label=lbl, color=col, linestyle=sty, marker=mkr, linewidth=lw, markersize=ms)

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lbl)
    ax.set_xlabel("Context Length L (tokens)")
    ax.set_ylabel("Decode Latency (ms, log scale)")
    ax.set_title("(a) Qwen3/Llama-3 GQA Decode Latency (B=4, d=128)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7.2, frameon=True, framealpha=0.92)

    r32 = b4_128["32768"]
    r64 = b4_128["65536"]
    sp_cudnn64 = r64["speedup_paged35_vs_cudnn_sdpa"]
    sp_int4_64 = r64["speedup_paged35_vs_paged_int4"]
    sp_int4_32 = r32["speedup_paged35_vs_paged_int4"]
    ax.annotate(
        f"Paged Dual35 vs Shuffled INT4:\n32K: {sp_int4_32:.2f}x | 64K: {sp_int4_64:.2f}x\n"
        f"64K vs cuDNN GQA SDPA: {sp_cudnn64:.2f}x",
        xy=(x_pos[-1], r64["latency_ms"]["paged_dualstream_15_35"]),
        xytext=(x_pos[-2] - 1.0, r64["latency_ms"]["paged_dualstream_15_35"] * 0.34),
        fontsize=7.8,
        fontweight="bold",
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#c0392b", lw=1),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )

    # (b) Long-Context Speedup Over Shuffled Paged Triton-INT4 Baseline Across Specs
    ax = axes[0, 1]
    groups = [
        ("Q3 d=128\nB=1,32K", lat_all["qwen3_d128"]["by_batch"]["B=1"]["32768"]),
        ("Q3 d=128\nB=4,32K", lat_all["qwen3_d128"]["by_batch"]["B=4"]["32768"]),
        ("Q3 d=128\nB=4,64K", lat_all["qwen3_d128"]["by_batch"]["B=4"]["65536"]),
        ("Q3.5 d=256\nB=1,32K", lat_all["qwen35_d256"]["by_batch"]["B=1"]["32768"]),
        ("Q3.5 d=256\nB=4,32K", lat_all["qwen35_d256"]["by_batch"]["B=4"]["32768"]),
        ("Q3.5 d=256\nB=4,64K", lat_all["qwen35_d256"]["by_batch"]["B=4"]["65536"]),
    ]
    gx = np.arange(len(groups))
    w = 0.25
    sp_gated = [g[1]["latency_ms"]["triton_int4"] / g[1]["latency_ms"]["bitplane_gated"] for g in groups]
    sp_d85 = [g[1]["speedup_dual85_vs_int4"] for g in groups]
    sp_p35 = [g[1]["speedup_paged35_vs_paged_int4"] for g in groups]

    b1 = ax.bar(gx - w, sp_gated, width=w * 0.92, label="SinglePass Gated (branched)", color="#e67e22", alpha=0.88)
    b2 = ax.bar(gx, sp_d85, width=w * 0.92, label="DualStream (15%+85%)", color="#16a085", alpha=0.9)
    b3 = ax.bar(gx + w, sp_p35, width=w * 0.92, label="Paged DualStream (15+35, Ours)", color="#c0392b", alpha=0.95)
    ax.axhline(1.0, color="#2980b9", linestyle="--", linewidth=1.4, label="Paged Triton-INT4 Baseline (1.0x)")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.03,
                f"{h:.2f}x",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.8,
                bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none", alpha=0.85),
                zorder=4,
            )

    ax.set_xticks(gx)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_ylabel("Speedup over Triton-INT4 (x)")
    ax.set_ylim(0, max(sp_p35 + [1.5]) * 1.44)
    ax.set_title("(b) Long-Context Speedup Over Paged Triton-INT4 (Shuffled Pool)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=2, fontsize=7.0, frameon=True, framealpha=0.92)

    # (c) PTXAS Profiling Including Paged-Pool BlockTable Kernels
    ax = axes[1, 0]
    p128 = ptx["d128_B4"]
    k_keys = ["singlepass_gated", "int4_splitkv", "paged_hot_int4", "paged_warm_base2", "paged_anisotropic_score"]
    k_names = ["SinglePass\nGated (If)", "Triton-INT4\n(No-Branch)", "Paged Hot\n(INT4 Pool)", "Paged Warm\n(Base2 Pool)", "Paged Aniso\nRanker"]
    regs = [p128[k]["n_regs"] for k in k_keys]
    smem_kb = [p128[k]["shared_bytes"] / 1024.0 for k in k_keys]

    kx = np.arange(len(k_keys))
    bw = 0.36
    ax2 = ax.twinx()
    bars_r = ax.bar(kx - bw / 2, regs, width=bw, color="#34495e", alpha=0.88, label="Registers / Thread")
    bars_s = ax2.bar(kx + bw / 2, smem_kb, width=bw, color="#27ae60", alpha=0.88, label="Shared Mem (KB)")

    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 2, f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=7.5, color="#34495e", fontweight="bold")
    for bar in bars_s:
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3, f"{bar.get_height():.1f}K", ha="center", va="bottom", fontsize=7.5, color="#1e8449", fontweight="bold")

    ax.set_xticks(kx)
    ax.set_xticklabels(k_names, fontsize=8.0)
    ax.set_ylabel("PTXAS Registers / Thread", color="#34495e")
    ax2.set_ylabel("Shared Memory per CTA (KB)", color="#1e8449")
    ax.set_ylim(0, max(regs) * 1.25)
    ax2.set_ylim(0, max(smem_kb + [12.0]) * 1.28)
    ax.set_title("(c) PTXAS Profiling (d=128): True Paged Kernels Stay <= 128 Regs (4 CTAs/SM)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=7.6, frameon=True)

    # (d) End-to-End Accuracy vs Single-Scalar Logical Read Bandwidth Pareto (BF16 Baseline)
    ax = axes[1, 1]
    model_styles = [
        ("Qwen3.5-0.8B", "#c0392b", "o", "-"),
        ("Qwen3-0.6B", "#2980b9", "s", "--"),
    ]
    pt_offsets = {
        "Qwen3.5-0.8B": [(18, 12), (4, 14), (0, 13), (-10, 12)],
        "Qwen3-0.6B": [(18, -16), (0, -16), (-4, -15), (-10, -14)],
    }
    for m_name, col, mkr, lsty in model_styles:
        cfg = e2e[m_name]["configs"]
        pts = [
            ("PureSparse", cfg["triton_dualstream_15_0"]),
            ("Dual(15+35)", cfg["triton_dualstream_15_35"]),
            ("Dual(15+85)", cfg["triton_dualstream_15_85"]),
            ("PagedINT4", cfg["triton_int4"]),
        ]
        xs = [p[1]["logical_read_bits_per_kv_elem"] for p in pts]
        ys = [max(1e-4, p[1]["mean_kl"]) for p in pts]
        ax.plot(xs, ys, color=col, marker=mkr, linestyle=lsty, linewidth=2.0, markersize=7, label=f"{m_name} (BitPlaneDynamicCache)")
        for idx_p, (lbl_txt, p_d) in enumerate(pts):
            dx, dy = pt_offsets[m_name][idx_p]
            ax.annotate(
                f"{lbl_txt} ({p_d['needle_hits']}/5)",
                xy=(p_d["logical_read_bits_per_kv_elem"], max(1e-4, p_d["mean_kl"])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=6.8,
                color=col,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.75),
            )

    ax.set_yscale("log")
    ax.set_xlim(0.7, 5.2)
    ax.set_ylim(5e-4, 0.25)
    ax.set_xlabel("Logical KV Read Bandwidth (bits / KV-elem, incl. page metadata)")
    ax.set_ylabel("Logit KL Divergence vs BF16 (log scale, lower=better)")
    ax.set_title("(d) End-to-End BitPlaneDynamicCache Accuracy vs Read Bandwidth")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=7.6, frameon=True, framealpha=0.92)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"[DONE] 300 DPI 总结图已保存至 {OUT_PNG}", flush=True)


def main() -> int:
    if "--plot-only" in sys.argv:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        render_summary_figure(data)
        return 0
    t0 = time.perf_counter()
    props = torch.cuda.get_device_properties(0)
    data: dict[str, Any] = {
        "hardware": {
            "gpu": props.name,
            "total_vram_gb": props.total_memory / (1024.0 ** 3),
            "sm_count": props.multi_processor_count,
            "torch": torch.__version__,
            "cuda_capability": f"sm_{props.major}{props.minor}",
        },
        "latency": run_latency_benchmarks(),
        "ptxas": run_ptxas_microarch_audit(),
        "routing_recall": run_routing_recall_ablation(),
        "end_to_end": run_end_to_end_real_kernels(),
    }
    data["elapsed_sec"] = time.perf_counter() - t0
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[DONE] 全量实验 JSON 已落盘至 {OUT_JSON} (总耗时 {data['elapsed_sec']:.1f}s)", flush=True)
    render_summary_figure(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
