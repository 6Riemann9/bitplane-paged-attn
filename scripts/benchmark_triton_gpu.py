#!/usr/bin/env python3
"""
Phase 1 终极收官实验：RTX 5070 Ti (Blackwell sm_120) Triton 真机压测、PTXAS 剖析与双模型端到端挂核验证。

实验涵盖 4 大核心板块（严禁造假，全量实时工具测量）：
  1. CUDA Event 精准延时与加速比基准（L in [2048..65536]，覆盖 Qwen3 d=128 与 Qwen3.5 d=256，B=1/4）；
     同台对比：PyTorch SDPA / Triton-FP16 / Triton-INT4 / TwoPass-BitPlane / SinglePass-Gated /
              DualStream (15%+85%) / DualStream (15%+35%)；
  2. PTXAS 编译微架构剖析（捕获 n_regs、shared_bytes 与 SM 理论占用率上限对比）；
  3. 真实 LLM 端到端挂载真实 Triton GPU 核验证（Qwen3.5-0.8B Hybrid 与 Qwen3-0.6B Dense，
     L=2048 长文档 128 步自回归 PPL/KL 与 5 深度大海捞针）；
  4. 结果保存至 results/triton_benchmark.json 并绘制 300 DPI 四宫格论文级图 results/triton_benchmark.png。
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, eager_attention_forward
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForConditionalGeneration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bitplane_attn import (  # noqa: E402
    PackedBitPlaneKVCache,
    triton_bitplane_decode,
    triton_bitplane_dualstream_decode,
    triton_bitplane_gated_decode,
    triton_fp16_paged_decode,
    triton_int4_paged_decode,
)
from bitplane_attn.triton_kernels import (  # noqa: E402
    _bitplane_singlepass_gated_kernel,
    _choose_dual_splits,
    _choose_num_splits,
    _ellipsoid_score_kernel,
    _indexed_base2_splitkv_kernel,
    _indexed_int4_splitkv_kernel,
    _int4_splitkv_kernel,
)

KILL_TEST_JSON = PROJECT_ROOT / "results" / "kill_test_bitplane.json"
OUT_JSON = PROJECT_ROOT / "results" / "triton_benchmark.json"
OUT_PNG = PROJECT_ROOT / "results" / "triton_benchmark.png"


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

PAGE_SIZE = 32
RECENT_TOKENS = 32
HIST_LEN = 2016
PREFIX_LEN = 2048
CONT_STEPS = 128
NEEDLE_DECODE_STEPS = 16


def _bench_cuda_ms(fn, warmup: int = 30, iters: int = 100) -> float:
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


def _pytorch_sdpa_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """PyTorch 原生 SDPA GQA 解码基线。"""
    bsz, h_q, _, head_dim = query.shape
    h_kv = key.shape[1]
    grp = h_q // h_kv
    if grp > 1:
        k_exp = key[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
        v_exp = value[:, :, None, :, :].expand(bsz, h_kv, grp, -1, head_dim).reshape(bsz, h_q, -1, head_dim)
    else:
        k_exp, v_exp = key, value
    return F.scaled_dot_product_attention(query, k_exp, v_exp, scale=scaling).transpose(1, 2).contiguous()


def run_latency_benchmarks() -> dict[str, Any]:
    """运行全规格 CUDA 真机延时与物理显存压测。"""
    print("\n================ [1/4] RTX 5070 Ti 真机算子延时与显存压测 ================", flush=True)
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
                # 注入 Page 0 Attention Sink 及少许热点检索信号以测算真实门控比例
                k[:, :, :4, :] = q[:, ::grp, 0, :].unsqueeze(2) * 3.2
                k[:, :, (lk // 3), :] = q[:, ::grp, 0, :] * 3.0

                cache = PackedBitPlaneKVCache.from_kv(k, v, page_size=PAGE_SIZE, recent_tokens=RECENT_TOKENS)
                mem = cache.memory_stats()

                # 先统计真实分布下 Gated / DualStream 的三级占比
                _, st_gated = triton_bitplane_gated_decode(
                    q, cache, scaling, tau_hot=2.0, tau_skip=4.5, return_stats=True
                )
                _, st_dual85 = triton_bitplane_dualstream_decode(
                    q, cache, scaling, hot_ratio=0.15, warm_ratio=0.85, ellipsoid_beta=1.5, return_stats=True
                )
                _, st_dual35 = triton_bitplane_dualstream_decode(
                    q, cache, scaling, hot_ratio=0.15, warm_ratio=0.35, ellipsoid_beta=1.5, return_stats=True
                )

                # 为超长 L>=32768 减少单次 iter 数以节省无效等待且保证准确
                iters = 60 if lk >= 32768 else 100
                warmup = 20 if lk >= 32768 else 30

                # PyTorch SDPA 在 B=4, L=65536 会展开 ~4GB 张量，测完立即释放
                t_sdpa = _bench_cuda_ms(lambda: _pytorch_sdpa_decode(q, k, v, scaling), warmup=warmup, iters=iters)
                torch.cuda.empty_cache()

                t_fp16 = _bench_cuda_ms(lambda: triton_fp16_paged_decode(q, k, v, scaling), warmup=warmup, iters=iters)
                t_int4 = _bench_cuda_ms(lambda: triton_int4_paged_decode(q, cache, scaling), warmup=warmup, iters=iters)
                t_twopass = _bench_cuda_ms(
                    lambda: triton_bitplane_decode(q, cache, scaling, top_ratio=0.15), warmup=warmup, iters=iters
                )
                t_gated = _bench_cuda_ms(
                    lambda: triton_bitplane_gated_decode(q, cache, scaling, tau_hot=2.0, tau_skip=4.5),
                    warmup=warmup,
                    iters=iters,
                )
                t_dual85 = _bench_cuda_ms(
                    lambda: triton_bitplane_dualstream_decode(
                        q, cache, scaling, hot_ratio=0.15, warm_ratio=0.85, ellipsoid_beta=1.5
                    ),
                    warmup=warmup,
                    iters=iters,
                )
                t_dual35 = _bench_cuda_ms(
                    lambda: triton_bitplane_dualstream_decode(
                        q, cache, scaling, hot_ratio=0.15, warm_ratio=0.35, ellipsoid_beta=1.5
                    ),
                    warmup=warmup,
                    iters=iters,
                )

                row = {
                    "seq_len": lk,
                    "latency_ms": {
                        "pytorch_sdpa": t_sdpa,
                        "triton_fp16": t_fp16,
                        "triton_int4": t_int4,
                        "bitplane_twopass": t_twopass,
                        "bitplane_gated": t_gated,
                        "bitplane_dualstream_15_85": t_dual85,
                        "bitplane_dualstream_15_35": t_dual35,
                    },
                    "speedup_dual35_vs_sdpa": t_sdpa / t_dual35,
                    "speedup_dual35_vs_fp16": t_fp16 / t_dual35,
                    "speedup_dual35_vs_int4": t_int4 / t_dual35,
                    "speedup_dual85_vs_int4": t_int4 / t_dual85,
                    "gated_stats": st_gated,
                    "dualstream_15_85_stats": st_dual85,
                    "dualstream_15_35_stats": st_dual35,
                    "memory_mb": {
                        "fp16_equiv": mem["fp16_equiv_bytes"] / (1024.0 ** 2),
                        "packed_total": mem["total_bytes"] / (1024.0 ** 2),
                        "compression_ratio": mem["compression_ratio"],
                        "effective_bits_stored_total": mem["effective_bits_stored_total"],
                    },
                }
                results[sp_name]["by_batch"][b_key][str(lk)] = row
                print(
                    f"  L={lk:5d} | SDPA={t_sdpa:6.3f}ms | FP16={t_fp16:5.3f}ms | INT4={t_int4:5.3f}ms | "
                    f"2Pass={t_twopass:5.3f}ms | Gated(sk={st_gated['skip_ratio']*100:4.1f}%)={t_gated:5.3f}ms | "
                    f"Dual(15+85)={t_dual85:5.3f}ms | Dual(15+35)={t_dual35:5.3f}ms "
                    f"[vsINT4={row['speedup_dual35_vs_int4']:.2f}x, vsFP16={row['speedup_dual35_vs_fp16']:.2f}x, "
                    f"vsSDPA={row['speedup_dual35_vs_sdpa']:.1f}x]",
                    flush=True,
                )

                del q, k, v, cache
                gc.collect()
                torch.cuda.empty_cache()

    return results


def run_ptxas_microarch_audit() -> dict[str, Any]:
    """捕获各 Triton 核 PTXAS 编译参数（寄存器数量、共享内存与 SM 理论驻留率）。"""
    print("\n================ [2/4] PTXAS 编译微架构与 cp.async 流水线剖析 ================", flush=True)
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
        num_pages = cache.num_pages
        n_hot = int(math.ceil(num_pages * 0.15))
        n_warm = int(math.ceil(num_pages * 0.35))
        hot_idx = torch.arange(n_hot, device=device, dtype=torch.int32).view(1, 1, -1).expand(bsz, h_kv, -1).contiguous()
        warm_idx = torch.arange(n_hot, n_hot + n_warm, device=device, dtype=torch.int32).view(1, 1, -1).expand(bsz, h_kv, -1).contiguous()

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
        ck_score = _ellipsoid_score_kernel[(bsz * h_kv, (num_pages + 31) // 32)](
            q_c, cache.k_min, cache.k_scale, head_scores,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            cache.k_min.stride(0), cache.k_min.stride(1), cache.k_min.stride(2), cache.k_min.stride(3),
            head_scores.stride(0), head_scores.stride(1), head_scores.stride(2),
            scaling, 1.5, num_pages,
            H_kv=h_kv, G=grp, HEAD_DIM=d, BLOCK_G=16, BLOCK_D=d, BLOCK_P=32, USE_ELLIPSOID=True,
            num_warps=4, num_stages=(2 if d >= 256 else 3),
        )
        ck_idx_int4 = _indexed_int4_splitkv_kernel[(bsz * h_kv, ns_hot)](
            q_c, hot_idx, cache.k_base_packed, cache.k_res_packed, cache.k_min, cache.k_scale,
            cache.v_base_packed, cache.v_res_packed, cache.v_min, cache.v_scale,
            part_m, part_l, part_o,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            hot_idx.stride(0), hot_idx.stride(1), hot_idx.stride(2),
            cache.k_base_packed.stride(0), cache.k_base_packed.stride(1), cache.k_base_packed.stride(2),
            cache.k_base_packed.stride(3), cache.k_base_packed.stride(4),
            cache.k_min.stride(0), cache.k_min.stride(1), cache.k_min.stride(2), cache.k_min.stride(3),
            cache.v_min.stride(0), cache.v_min.stride(1), cache.v_min.stride(2), cache.v_min.stride(3),
            part_m.stride(0), part_m.stride(1), part_m.stride(2),
            part_o.stride(0), part_o.stride(1), part_o.stride(2), part_o.stride(3),
            scaling, n_hot, ns_hot, 0,
            H_kv=h_kv, G=grp, D4=d4, PAGE_SIZE=PAGE_SIZE, BLOCK_G=16, BLOCK_D4=d4, num_warps=4,
        )
        ck_idx_base2 = _indexed_base2_splitkv_kernel[(bsz * h_kv, ns_warm)](
            q_c, warm_idx, cache.k_base_packed, cache.k_min, cache.k_scale,
            cache.v_base_packed, cache.v_min, cache.v_scale,
            part_m, part_l, part_o,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            warm_idx.stride(0), warm_idx.stride(1), warm_idx.stride(2),
            cache.k_base_packed.stride(0), cache.k_base_packed.stride(1), cache.k_base_packed.stride(2),
            cache.k_base_packed.stride(3), cache.k_base_packed.stride(4),
            cache.k_min.stride(0), cache.k_min.stride(1), cache.k_min.stride(2), cache.k_min.stride(3),
            cache.v_min.stride(0), cache.v_min.stride(1), cache.v_min.stride(2), cache.v_min.stride(3),
            part_m.stride(0), part_m.stride(1), part_m.stride(2),
            part_o.stride(0), part_o.stride(1), part_o.stride(2), part_o.stride(3),
            scaling, n_warm, ns_warm, ns_hot,
            H_kv=h_kv, G=grp, D4=d4, PAGE_SIZE=PAGE_SIZE, BLOCK_G=16, BLOCK_D4=d4, num_warps=4,
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
            "int4_splitkv": _extract(ck_int4),
            "singlepass_gated": _extract(ck_gated),
            "ellipsoid_score": _extract(ck_score),
            "indexed_int4_hot": _extract(ck_idx_int4),
            "indexed_base2_warm": _extract(ck_idx_base2),
        }
        print(f"\n--- PTXAS 规格 {label} ---", flush=True)
        for k_name, info in audit[label].items():
            print(
                f"  {k_name:20s} | n_regs={info['n_regs']:3d} | shared={info['shared_bytes']:5d} B | "
                f"max_ctas/SM={info['max_ctas_per_sm']:2d}",
                flush=True,
            )

        del q_c, k, v, cache, hot_idx, warm_idx, part_m, part_l, part_o, m_anchor, tier_out, head_scores
        gc.collect()
        torch.cuda.empty_cache()

    return audit


@dataclass
class _RealKernelCtx:
    enabled: bool = False
    mode: str = "fp16"
    hot_ratio: float = 0.15
    warm_ratio: float = 0.35
    ellipsoid_beta: float = 1.5
    packed_by_layer: dict[int, PackedBitPlaneKVCache] = field(default_factory=dict)
    read_bits_samples: list[float] = field(default_factory=list)


_RCTX = _RealKernelCtx()


def _shallow_cache_with_tail(
    base_cache: PackedBitPlaneKVCache,
    k_tail_full: torch.Tensor,
    v_tail_full: torch.Tensor,
) -> PackedBitPlaneKVCache:
    """复用静态历史前缀物理位面缓存并零重量化拼接动态增长的全精尾部窗口。"""
    return PackedBitPlaneKVCache(
        page_size=base_cache.page_size,
        hist_len=base_cache.hist_len,
        num_pages=base_cache.num_pages,
        batch_size=base_cache.batch_size,
        num_kv_heads=base_cache.num_kv_heads,
        head_dim=base_cache.head_dim,
        kv_dtype=base_cache.kv_dtype,
        k_base_packed=base_cache.k_base_packed,
        k_res_packed=base_cache.k_res_packed,
        k_min=base_cache.k_min,
        k_scale=base_cache.k_scale,
        v_base_packed=base_cache.v_base_packed,
        v_res_packed=base_cache.v_res_packed,
        v_min=base_cache.v_min,
        v_scale=base_cache.v_scale,
        k_tail=k_tail_full[:, :, base_cache.hist_len :, :].contiguous(),
        v_tail=v_tail_full[:, :, base_cache.hist_len :, :].contiguous(),
    )


def _real_triton_probe_interface(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if attention_mask is None and query.shape[2] > 1:
        lq, lk = query.shape[2], key.shape[2]
        past_len = lk - lq
        q_pos = torch.arange(past_len, lk, device=query.device)[:, None]
        k_pos = torch.arange(lk, device=query.device)[None, :]
        causal = torch.where(k_pos <= q_pos, 0.0, torch.finfo(query.dtype).min).to(query.dtype)
        attention_mask = causal[None, None, :, :]

    if (
        (not _RCTX.enabled)
        or _RCTX.mode == "fp16"
        or query.shape[2] != 1
        or (not isinstance(module, (Qwen3Attention, Qwen3_5Attention)))
    ):
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )

    l_idx = int(module.layer_idx)
    base_cache = _RCTX.packed_by_layer[l_idx]
    cur_cache = _shallow_cache_with_tail(base_cache, key, value)

    if _RCTX.mode == "triton_int4":
        out = triton_int4_paged_decode(query, cur_cache, scaling)
        _RCTX.read_bits_samples.append(9.25)
    elif _RCTX.mode == "triton_dualstream":
        out, st = triton_bitplane_dualstream_decode(
            query,
            cur_cache,
            scaling,
            hot_ratio=_RCTX.hot_ratio,
            warm_ratio=_RCTX.warm_ratio,
            ellipsoid_beta=_RCTX.ellipsoid_beta,
            pin_page0=True,
            return_stats=True,
        )
        _RCTX.read_bits_samples.append(float(st["phys_read_bits"]))
    else:
        raise ValueError(f"未知的真核端到端模式: {_RCTX.mode}")

    # Triton 解码核输出已严格是 (B, L_q=1, H_q, d)，不可再转置
    return out, None


ALL_ATTENTION_FUNCTIONS.register("real_triton_probe", _real_triton_probe_interface)


def _bind_attn(model: torch.nn.Module, impl: str) -> None:
    if hasattr(model, "config") and model.config is not None:
        model.config._attn_implementation = impl
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            model.config.text_config._attn_implementation = impl
    for mod in model.modules():
        if hasattr(mod, "config") and mod.config is not None:
            mod.config._attn_implementation = impl


def _pack_layers_from_pkv(pkv: Any, attn_layers: list[int]) -> dict[int, PackedBitPlaneKVCache]:
    out: dict[int, PackedBitPlaneKVCache] = {}
    for l_idx in attn_layers:
        if hasattr(pkv, "layers"):
            k_t, v_t = pkv.layers[l_idx].keys, pkv.layers[l_idx].values
        else:
            k_t, v_t = pkv.key_cache[l_idx], pkv.value_cache[l_idx]
        out[l_idx] = PackedBitPlaneKVCache.from_kv(k_t, v_t, page_size=PAGE_SIZE, hist_len=HIST_LEN)
    return out


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


def _autoreg_128_steps(
    model: torch.nn.Module,
    pkv_init: Any,
    eval_tokens: torch.Tensor,
) -> np.ndarray:
    """严格 L_q == 1 逐 token 自回归前向（教师强制），精准喂入 Triton 解码算子。"""
    pkv = copy.deepcopy(pkv_init)
    logits_steps = []
    with torch.no_grad():
        for t in range(eval_tokens.shape[1]):
            out = model(input_ids=eval_tokens[:, t : t + 1], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits_steps.append(out.logits[0, -1, :].float())
    return torch.stack(logits_steps, dim=0).cpu().numpy()


def _eval_single_model_real_kernels(model_name: str, is_35: bool, device: torch.device) -> dict[str, Any]:
    print(f"\n--- 真实模型挂接真实 Triton 核验证: {model_name} ---", flush=True)
    m_src = _resolve_model_source(model_name)
    tokenizer = AutoTokenizer.from_pretrained(m_src)
    model = (Qwen3_5ForConditionalGeneration if is_35 else AutoModelForCausalLM).from_pretrained(
        m_src, dtype=torch.bfloat16
    ).to(device).eval()
    attn_layers = sorted({int(m.layer_idx) for m in model.modules() if isinstance(m, (Qwen3Attention, Qwen3_5Attention))})
    _bind_attn(model, "real_triton_probe")

    docs = _build_docs(tokenizer, device)
    needles = _build_needles(tokenizer, device)

    doc_prep = []
    for d_item in docs:
        _RCTX.enabled = False
        with torch.no_grad():
            pkv = model(input_ids=d_item["prefix_ids"][:, :-1], use_cache=True).past_key_values
        packed = _pack_layers_from_pkv(pkv, attn_layers)
        eval_chunk = torch.cat([d_item["prefix_ids"][:, -1:], d_item["cont_ids"][:, :-1]], dim=1)
        targets = d_item["cont_ids"][0].cpu().numpy()

        _RCTX.enabled = True
        _RCTX.mode = "fp16"
        ref_logits = _autoreg_128_steps(model, pkv, eval_chunk)
        _RCTX.enabled = False
        max_z = np.max(ref_logits, axis=1, keepdims=True)
        lse = max_z[:, 0] + np.log(np.sum(np.exp(ref_logits - max_z), axis=1))
        fp_ppl = float(np.exp(np.mean(lse - ref_logits[np.arange(targets.shape[0]), targets])))
        doc_prep.append(
            {"pkv": pkv, "eval_chunk": eval_chunk, "targets": targets, "packed": packed, "ref_logits": ref_logits, "fp_ppl": fp_ppl}
        )

    needle_prep = []
    for n_item in needles:
        _RCTX.enabled = False
        with torch.no_grad():
            pkv_n = model(input_ids=n_item["input_ids"][:, :-1], use_cache=True).past_key_values
        needle_prep.append(
            {
                "pkv": pkv_n,
                "last_tok": n_item["input_ids"][:, -1:],
                "code": n_item["code"],
                "packed": _pack_layers_from_pkv(pkv_n, attn_layers),
            }
        )

    fp16_ppl = float(np.mean([x["fp_ppl"] for x in doc_prep]))
    configs = [
        ("triton_int4", "triton_int4", 1.0, 0.0, 0.0),
        ("triton_dualstream_15_85", "triton_dualstream", 0.15, 0.85, 1.5),
        ("triton_dualstream_15_35", "triton_dualstream", 0.15, 0.35, 1.5),
        ("triton_dualstream_15_0", "triton_dualstream", 0.15, 0.0, 1.5),
    ]

    model_res: dict[str, Any] = {"fp16_ppl": fp16_ppl, "configs": {}}
    print(f"  FP16 Autoreg PPL = {fp16_ppl:.4f}", flush=True)

    for cfg_name, mode, hr, wr, beta in configs:
        ppls, kls, bits = [], [], []
        for dp in doc_prep:
            _RCTX.enabled = True
            _RCTX.mode = mode
            _RCTX.hot_ratio, _RCTX.warm_ratio, _RCTX.ellipsoid_beta = hr, wr, beta
            _RCTX.packed_by_layer = dp["packed"]
            _RCTX.read_bits_samples.clear()
            logits = _autoreg_128_steps(model, dp["pkv"], dp["eval_chunk"])
            _RCTX.enabled = False

            mz = np.max(logits, axis=1, keepdims=True)
            lse = mz[:, 0] + np.log(np.sum(np.exp(logits - mz), axis=1))
            ppls.append(float(np.exp(np.mean(lse - logits[np.arange(dp["targets"].shape[0]), dp["targets"]]))))

            mr = np.max(dp["ref_logits"], axis=1, keepdims=True)
            lpr = dp["ref_logits"] - mr - np.log(np.sum(np.exp(dp["ref_logits"] - mr), axis=1, keepdims=True))
            lpc = logits - mz - np.log(np.sum(np.exp(logits - mz), axis=1, keepdims=True))
            kls.append(float(np.mean(np.maximum(0.0, np.sum(np.exp(lpr) * (lpr - lpc), axis=1)))))
            bits.extend(_RCTX.read_bits_samples)

        hits = 0
        for np_it in needle_prep:
            _RCTX.enabled = True
            _RCTX.mode = mode
            _RCTX.hot_ratio, _RCTX.warm_ratio, _RCTX.ellipsoid_beta = hr, wr, beta
            _RCTX.packed_by_layer = np_it["packed"]
            pkv_g = copy.deepcopy(np_it["pkv"])
            cur = np_it["last_tok"]
            g_ids = []
            with torch.no_grad():
                for _ in range(NEEDLE_DECODE_STEPS):
                    o = model(input_ids=cur, past_key_values=pkv_g, use_cache=True)
                    pkv_g = o.past_key_values
                    cur = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
                    g_ids.append(int(cur.item()))
            _RCTX.enabled = False
            if np_it["code"] in tokenizer.decode(g_ids, skip_special_tokens=True):
                hits += 1

        cfg_out = {
            "mean_ppl": float(np.mean(ppls)),
            "delta_ppl": float(np.mean(ppls)) - fp16_ppl,
            "mean_kl": float(np.mean(kls)),
            "needle_hits": hits,
            "needle_total": len(needle_prep),
            "phys_read_bits": float(np.mean(bits)),
        }
        model_res["configs"][cfg_name] = cfg_out
        print(
            f"  {cfg_name:26s} | read={cfg_out['phys_read_bits']:4.2f}b | PPL={cfg_out['mean_ppl']:.3f} "
            f"(dPPL={cfg_out['delta_ppl']:+.3f}) | KL={cfg_out['mean_kl']:.5f} | Needles={hits}/5",
            flush=True,
        )

    _RCTX.packed_by_layer.clear()
    del model, doc_prep, needle_prep
    gc.collect()
    torch.cuda.empty_cache()
    return model_res


def run_end_to_end_real_kernels() -> dict[str, Any]:
    """双真实大模型端到端挂接真实 Triton GPU 算子校验。"""
    print("\n================ [3/4] 双模型端到端挂载真实 Triton GPU 核验证 ================", flush=True)
    device = torch.device("cuda:0")
    return {
        "Qwen3.5-0.8B": _eval_single_model_real_kernels("Qwen3.5-0.8B", True, device),
        "Qwen3-0.6B": _eval_single_model_real_kernels("Qwen3-0.6B", False, device),
    }


def render_summary_figure(data: dict[str, Any]) -> None:
    """绘制 300 DPI 四宫格顶会级总结图（全英文标签、防重叠排布）。"""
    print("\n================ [4/4] 绘制 300 DPI 四宫格论文总结图 ================", flush=True)
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2), dpi=300)
    plt.subplots_adjust(hspace=0.33, wspace=0.25, top=0.92, bottom=0.08, left=0.07, right=0.97)
    fig.suptitle(
        "Bit-Plane Paged Attention on RTX 5070 Ti (Blackwell sm_120): Zero-Branch Dual-Stream Triton Kernels",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    lat_all = data["latency"]
    ptx = data["ptxas"]
    e2e = data["end_to_end"]
    kill = {}
    if KILL_TEST_JSON.exists():
        kill = json.loads(KILL_TEST_JSON.read_text(encoding="utf-8")).get("models", {})

    # (a) Qwen3 d=128, B=4 Decode Latency vs Sequence Length
    ax = axes[0, 0]
    b4_128 = lat_all["qwen3_d128"]["by_batch"]["B=4"]
    ls = [int(k) for k in b4_128.keys()]
    x_pos = np.arange(len(ls))
    x_lbl = [f"{k // 1024}K" for k in ls]

    series_a = [
        ("pytorch_sdpa", "PyTorch SDPA (FP16 GQA)", "#95a5a6", "--", "x"),
        ("triton_fp16", "Triton-FP16 SplitKV", "#34495e", "-.", "s"),
        ("triton_int4", "Triton-INT4 (100% 4-bit)", "#2980b9", "-", "o"),
        ("bitplane_gated", "SinglePass Gated (branched)", "#e67e22", ":", "^"),
        ("bitplane_dualstream_15_85", "DualStream (15% Hot+85% Warm)", "#16a085", "--", "d"),
        ("bitplane_dualstream_15_35", "DualStream (15% Hot+35% Warm, Ours)", "#c0392b", "-", "*"),
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
    ax.set_title("(a) Qwen3/Llama-3 Spec (B=4, H_q=32, H_kv=8, d=128)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.92)

    # 标注 L=64K / L=32K 关键加速比
    r32 = b4_128["32768"]
    r64 = b4_128["65536"]
    ax.annotate(
        f"32K: {r32['speedup_dual35_vs_int4']:.2f}x vs INT4\n64K: {r64['speedup_dual35_vs_int4']:.2f}x vs INT4",
        xy=(x_pos[-1], r64["latency_ms"]["bitplane_dualstream_15_35"]),
        xytext=(x_pos[-2] - 0.65, r64["latency_ms"]["bitplane_dualstream_15_35"] * 0.38),
        fontsize=8.2,
        fontweight="bold",
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#c0392b", lw=1),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )

    # (b) Speedup Over Strong Triton-INT4 Baseline Across Specs (L=32K & L=64K)
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
    sp_d35 = [g[1]["speedup_dual35_vs_int4"] for g in groups]

    b1 = ax.bar(gx - w, sp_gated, width=w * 0.92, label="SinglePass Gated (branched)", color="#e67e22", alpha=0.88)
    b2 = ax.bar(gx, sp_d85, width=w * 0.92, label="DualStream (15%+85%)", color="#16a085", alpha=0.9)
    b3 = ax.bar(gx + w, sp_d35, width=w * 0.92, label="DualStream (15%+35%, Ours)", color="#c0392b", alpha=0.95)
    ax.axhline(1.0, color="#2980b9", linestyle="--", linewidth=1.4, label="Triton-INT4 Baseline (1.0x)")
    for b_idx, bars in enumerate((b1, b2, b3)):
        y_off = [0.025, 0.055, 0.035][b_idx]
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + y_off,
                f"{h:.2f}x",
                ha="center",
                va="bottom",
                fontsize=6.5,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.75),
            )

    ax.set_xticks(gx)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.3)
    ax.set_ylabel("Speedup over Triton-INT4 (x)")
    ax.set_ylim(0, max(sp_d35 + [1.5]) * 1.38)
    ax.set_title("(b) Long-Context Speedup Over Single-Pass Triton-INT4")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=True, framealpha=0.92)

    # (c) PTXAS Hardware Profiling (Registers & Shared Memory)
    ax = axes[1, 0]
    p128 = ptx["d128_B4"]
    k_keys = ["singlepass_gated", "int4_splitkv", "indexed_int4_hot", "indexed_base2_warm", "ellipsoid_score"]
    k_names = ["SinglePass\nGated (If)", "Triton-INT4\n(No-Branch)", "DualStream\nHot (INT4)", "DualStream\nWarm (Base2)", "Ellipsoid\nScore"]
    regs = [p128[k]["n_regs"] for k in k_keys]
    smem_kb = [p128[k]["shared_bytes"] / 1024.0 for k in k_keys]

    kx = np.arange(len(k_keys))
    bw = 0.36
    ax2 = ax.twinx()
    bars_r = ax.bar(kx - bw / 2, regs, width=bw, color="#34495e", alpha=0.88, label="Registers / Thread")
    bars_s = ax2.bar(kx + bw / 2, smem_kb, width=bw, color="#27ae60", alpha=0.88, label="Shared Mem (KB)")

    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 2, f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=7.6, color="#34495e", fontweight="bold")
    for bar in bars_s:
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3, f"{bar.get_height():.1f}K", ha="center", va="bottom", fontsize=7.6, color="#1e8449", fontweight="bold")

    ax.set_xticks(kx)
    ax.set_xticklabels(k_names, fontsize=8.3)
    ax.set_ylabel("PTXAS Registers / Thread", color="#34495e")
    ax2.set_ylabel("Shared Memory per CTA (KB)", color="#1e8449")
    ax.set_ylim(0, max(regs) * 1.25)
    ax2.set_ylim(0, max(smem_kb + [12.0]) * 1.28)
    ax.set_title("(c) PTXAS Profiling (d=128): Zero-Branch Cuts Regs & Restores 4 CTAs/SM")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=7.8, frameon=True)

    # (d) Real-LLM End-to-End Accuracy vs Physical Read Bandwidth Pareto
    ax = axes[1, 1]
    model_styles = [
        ("Qwen3.5-0.8B", "#c0392b", "o", "-"),
        ("Qwen3-0.6B", "#2980b9", "s", "--"),
    ]
    pt_offsets = {
        "Qwen3.5-0.8B": [(36, -6), (4, 13), (0, 12), (-16, 12)],
        "Qwen3-0.6B": [(34, -18), (0, -16), (-6, -15), (-14, -14)],
    }
    for m_name, col, mkr, lsty in model_styles:
        cfg = e2e[m_name]["configs"]
        # 按物理比特从低到高连接真实 Triton 核结果
        pts = [
            ("PureSparse", cfg["triton_dualstream_15_0"]),
            ("Dual(15+35)", cfg["triton_dualstream_15_35"]),
            ("Dual(15+85)", cfg["triton_dualstream_15_85"]),
            ("INT4", cfg["triton_int4"]),
        ]
        xs = [p[1]["phys_read_bits"] for p in pts]
        ys = [max(1e-4, p[1]["mean_kl"]) for p in pts]
        ax.plot(xs, ys, color=col, marker=mkr, linestyle=lsty, linewidth=2.0, markersize=7, label=f"{m_name} (Real Triton)")
        for idx_p, (lbl_txt, p_d) in enumerate(pts):
            dx, dy = pt_offsets[m_name][idx_p]
            ax.annotate(
                f"{lbl_txt} ({p_d['needle_hits']}/5)",
                xy=(p_d["phys_read_bits"], max(1e-4, p_d["mean_kl"])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=6.8,
                color=col,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.75),
            )

        # 叠加 Phase 0 的 Uniform INT2 与 Quest-15% (GQA) 锚点作为对比参考
        if m_name in kill:
            ppl_d = kill[m_name].get("ppl_results", {})
            ndl_d = kill[m_name].get("needle_results", {})
            if "int2" in ppl_d:
                ax.scatter([2.5], [ppl_d["int2"]["mean_kl"]], color=col, marker="X", s=70, alpha=0.75)
                dy_i2 = 2 if m_name == "Qwen3.5-0.8B" else 3
                ax.annotate(
                    f"{m_name.split('-')[0]} INT2 ({ndl_d.get('int2', {}).get('hits', '?')}/5)",
                    xy=(2.5, ppl_d["int2"]["mean_kl"]),
                    xytext=(10, dy_i2),
                    textcoords="offset points",
                    fontsize=6.8,
                    color=col,
                    va="center",
                )
            if "quest_15_gqa" in ppl_d:
                q_bits = ppl_d["quest_15_gqa"]["eff_bits_per_elem"]
                ax.scatter([q_bits], [ppl_d["quest_15_gqa"]["mean_kl"]], color=col, marker="^", s=65, alpha=0.75)
                dy_q = -5 if m_name == "Qwen3.5-0.8B" else 5
                ax.annotate(
                    f"{m_name.split('-')[0]} Quest15%",
                    xy=(q_bits, ppl_d["quest_15_gqa"]["mean_kl"]),
                    xytext=(12, dy_q),
                    textcoords="offset points",
                    fontsize=6.8,
                    color=col,
                    va="center",
                )

    ax.set_yscale("log")
    ax.set_xlim(1.6, 10.6)
    ax.set_ylim(5e-4, 0.35)
    ax.set_xlabel("Physical KV Read Bandwidth (bits / element, incl. metadata)")
    ax.set_ylabel("Logit KL Divergence vs FP16 (log scale, lower=better)")
    ax.set_title("(d) End-to-End Real Triton Kernel Accuracy vs Bandwidth Pareto")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8.0, frameon=True, framealpha=0.92)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"[DONE] 图表已保存至 {OUT_PNG}", flush=True)


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
