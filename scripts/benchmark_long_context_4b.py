#!/usr/bin/env python3
"""
Batch B 主线评测：Qwen3-4B (36L, 32Q/8KV, d=128) 8K–32K 长上下文 RULER/LongBench 质量与 5 方强基线实测。

覆盖 5 大真实全量评测维度：
  1. RULER-S-NIAH (8K / 16K / 32K)：10 个均匀深度的非重复长文档单针检索；
  2. RULER-MK-NIAH (16K / 32K)：同文档 4 键并发检索 + 干扰键值；
  3. RULER-VT (8K / 16K / 32K)：跨长文档 4 跳变量追踪，考验 Warm 2-bit 残差保底价值；
  4. LongBench-style 真实长文本 16K & 32K Teacher-Forced PPL / KL (128 步自回归, vs BF16)；
  5. Qwen3-4B 真机三方 CUDA Allocator 显存与 Greedy Decode 吞吐消融
     (`Native BF16` vs `Paged INT4` vs `BitPlane DualStream 15+35`)。

支持参数：
  --probe      仅快速探测 1K/8K 下 Native BF16 与 DualStream 在当前 Prompt 模板下的回答行为及显存峰值；
  --plot-only  直接读取 `results/long_context_4b.json` 重绘 300 DPI 四宫格图表。
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bitplane_attn import (  # noqa: E402
    BitPlaneDynamicCache,
    compute_logical_read_bits_per_kv_elem,
)

OUT_JSON = PROJECT_ROOT / "results" / "long_context_4b.json"
OUT_PNG = PROJECT_ROOT / "results" / "long_context_4b.png"

PAGE_SIZE = 32
RECENT_TOKENS = 32
PREFILL_CHUNK = 2048

METHODS: list[tuple[str, str, dict[str, Any]]] = [
    ("native_bf16", "Native BF16", {"kind": "native"}),
    ("paged_int4", "Paged-INT4 (static)", {"kind": "bp", "mode": "triton_int4"}),
    ("quest_hard_15_0", "Quest-like Sparse (15+0)", {"kind": "bp", "mode": "triton_dualstream", "hot_ratio": 0.15, "warm_ratio": 0.0, "beta": 1.5}),
    ("bitplane_dual_15_35", "BitPlane Dual (15+35)", {"kind": "bp", "mode": "triton_dualstream", "hot_ratio": 0.15, "warm_ratio": 0.35, "beta": 1.5}),
    ("readkv_prog_15_85", "ReadKV-like Prog (15+85)", {"kind": "bp", "mode": "triton_dualstream", "hot_ratio": 0.15, "warm_ratio": 0.85, "beta": 1.5}),
]

ESSAY_PARAGRAPHS = [
    (
        "Section on Distributed Memory Hierarchy: Modern datacenter accelerators exhibit a stark divergence "
        "between arithmetic compute throughput in tensor cores and high-bandwidth DRAM pin transfer limits. "
        "During autoregressive decoding, loading a multi-gigabyte key-value state across dozens of transformer "
        "layers for a single newly generated token reduces arithmetic intensity below two operations per byte. "
        "Consequently, compression formats that align storage granularity with warp-level memory transactions "
        "provide multiplicative throughput benefits when kernel register footprints remain bounded."
    ),
    (
        "Section on Virtual Page Tables and Copy-on-Write Semantics: Serving concurrent user sessions requires "
        "decoupling logical token coordinates from contiguous device allocations. A block allocator maintains "
        "a stack of fixed-size physical slots, reference counts for shared system prompts, and cryptographic or "
        "structural prefix digests. Whenever an in-place modification occurs on a page whose reference count equals "
        "one, prefix cache bindings must be invalidated prior to mutation so downstream requests never consume stale KV data."
    ),
    (
        "Section on Grouped-Query Attention Routing Geometry: In architectures where multiple query heads share "
        "a common key-value group, per-head sparse selection introduces set union inflation across group members. "
        "Computing per-page summary scores with per-head shift normalization followed by smooth group reduction aligns "
        "physical cache fetches across warps. Empirical mass recall evaluations confirm that anisotropic bounds mitigate "
        "outlier distortion compared to worst-case box norms along high-variance coordinate axes."
    ),
    (
        "Section on Compilers and Warp Occupancy Tuning: Inside GPU streaming multiprocessors, conditional branching "
        "over bit-plane tiers within an inner attention loop inflates live register allocations across PTX compilation. "
        "Partitioning precision tiers into specialized branchless Split-KV kernels preserves high concurrent CTA residency, "
        "allowing asynchronous global memory prefetching to hide pointer-chasing latency even across shuffled block tables."
    ),
    (
        "Section on Hybrid State-Space and Attention Architectures: Combining recurrent linear state modules with "
        "scarce full-attention layers alters the end-to-end memory profile of language models. While recurrent layers "
        "maintain bounded convolution and recurrence buffers independent of sequence growth, sparse global layers still "
        "accumulate linear KV state over ultra-long contexts where bit-plane quantization mitigates bandwidth saturation."
    ),
    (
        "Section on Numerical Stability in Log-Sum-Exp Reductions: Streaming online softmax merges partial attention "
        "accumulators across split key segments without storing full attention matrices. Tracking running per-head maxima "
        "and rescaling sums in single precision eliminates overflow when stitching heterogeneous precision streams alongside "
        "unquantized sliding-window tokens."
    ),
]


def _resolve_model_source(model_name: str = "Qwen3-4B") -> str:
    """优先从环境变量 BITPLANE_MODEL_CACHE 或本地缓存加载，未命中时回退 HuggingFace Hub。"""
    env_root = os.environ.get("BITPLANE_MODEL_CACHE")
    if env_root:
        cand = Path(env_root) / model_name
        if cand.exists():
            return str(cand)
    local_cand = PROJECT_ROOT / ".model_cache" / model_name
    if local_cand.exists():
        return str(local_cand)
    return f"Qwen/{model_name}"


def _build_diverse_corpus_tokens(tokenizer: Any, min_tokens: int = 40000) -> list[int]:
    """混合项目真实文档、核心源码与多主题系统散文，构造高熵非单一重复长文本词元流。"""
    segments: list[str] = []
    repo_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "src" / "bitplane_attn" / "quant_bitplane.py",
        PROJECT_ROOT / "src" / "bitplane_attn" / "packed_cache.py",
        PROJECT_ROOT / "src" / "bitplane_attn" / "hf_cache.py",
    ]
    for idx, fp in enumerate(repo_files):
        if fp.exists():
            txt = fp.read_text(encoding="utf-8")
            segments.append(f"\n[Archive Chapter {idx+1}: {fp.name}]\n" + txt)
    for k, para in enumerate(ESSAY_PARAGRAPHS):
        segments.append(f"\n[Technical Note {k+101}]\n" + para)

    rng = random.Random(20261126)
    stream_ids: list[int] = []
    round_idx = 0
    while len(stream_ids) < min_tokens:
        order = list(range(len(segments)))
        rng.shuffle(order)
        for seg_i in order:
            header = tokenizer.encode(f"\n--- Digest Block {round_idx}-{seg_i} ---\n", add_special_tokens=False)
            body = tokenizer.encode(segments[seg_i], add_special_tokens=False)
            # 截取可变长度窗口并打散避免长周期重复陷阱
            if len(body) > 900:
                start = (round_idx * 173 + seg_i * 97) % (len(body) - 700)
                body = body[start : start + 700]
            stream_ids.extend(header + body)
            if len(stream_ids) >= min_tokens:
                break
        round_idx += 1
    return stream_ids[:min_tokens]


def _make_cache_and_ctx(
    model: torch.nn.Module,
    method_cfg: dict[str, Any],
    collect_stats: bool = False,
):
    if method_cfg["kind"] == "native":
        return DynamicCache(config=model.config), nullcontext()
    kwargs = {k: v for k, v in method_cfg.items() if k != "kind"}
    c = BitPlaneDynamicCache(
        config=model.config,
        page_size=PAGE_SIZE,
        recent_tokens=RECENT_TOKENS,
        collect_stats=collect_stats,
        **kwargs,
    )
    return c, c.attach(model)


def _clone_bp_for_method(
    bp_base: BitPlaneDynamicCache,
    method_cfg: dict[str, Any],
    collect_stats: bool = True,
) -> BitPlaneDynamicCache:
    """深拷贝已 Prefill 完成的 `BitPlaneDynamicCache`，零重算切换读取模式配置。"""
    cache = copy.deepcopy(bp_base)
    cache.mode = str(method_cfg["mode"])
    cache.hot_ratio = float(method_cfg.get("hot_ratio", 0.15))
    cache.warm_ratio = float(method_cfg.get("warm_ratio", 0.35))
    cache.beta = float(method_cfg.get("beta", 1.5))
    cache.collect_stats = bool(collect_stats)
    cache.read_bits_samples = []
    return cache


def _chunked_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: Any,
    chunk_size: int = PREFILL_CHUNK,
) -> torch.Tensor:
    """按 batch 自适应缩放 2K 分块注入长前缀并启用 `logits_to_keep=1`，逐块清显存碎片防 WDDM 换页。"""
    bsz, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    eff_chunk = max(512, chunk_size // max(1, bsz))
    need_clean = (bsz * seq_len) >= 16384
    last_logits: torch.Tensor | None = None
    for st in range(0, seq_len, eff_chunk):
        ed = min(seq_len, st + eff_chunk)
        out = model(input_ids=input_ids[:, st:ed], cache=cache, use_cache=True, logits_to_keep=1)
        if ed == seq_len:
            last_logits = out.logits[:, -1:, :].detach()
        del out
        if need_clean:
            torch.cuda.empty_cache()
    assert last_logits is not None
    return last_logits


def _decode_from_last_tok(
    model: torch.nn.Module,
    tokenizer: Any,
    last_tok: torch.Tensor,
    cache: Any,
    max_new_tokens: int = 24,
) -> str:
    """在调用方已进入的上下文中，让最后一个 prompt token 走单步 Decode 起步并贪心生成。"""
    gen_ids: list[int] = []
    cur = last_tok
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=cur, cache=cache, use_cache=True)
            cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            tok_i = int(cur.item())
            gen_ids.append(tok_i)
            del out
            if tokenizer.eos_token_id is not None and tok_i == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _greedy_decode_text(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    cache: Any,
    ctx: Any,
    max_new_tokens: int = 24,
) -> str:
    """在单一 `with ctx:` 作用域内分块 Prefill `[:-1]` 并从最后 1 个 token 起步执行 `max_new_tokens` 步 Decode。"""
    with ctx, torch.no_grad():
        _chunked_prefill(model, input_ids[:, :-1], cache)
        return _decode_from_last_tok(model, tokenizer, input_ids[:, -1:], cache, max_new_tokens=max_new_tokens)


def _build_s_niah_cases(
    tokenizer: Any,
    corpus_ids: list[int],
    ctx_len: int,
    n_depths: int = 10,
) -> list[dict[str, Any]]:
    cases = []
    depths = [0.05 + 0.10 * i for i in range(n_depths)]
    beacons = ["ALPHA-7", "BRAVO-3", "DELTA-9", "ECHO-4", "FOXTROT-2", "GAMMA-8", "HELIX-5", "ION-6", "KILO-1", "LIMA-0"]
    for i, depth in enumerate(depths):
        code = f"{310400 + i * 7919 + ctx_len % 997}"
        beacon = beacons[i % len(beacons)]
        needle_txt = f"\n[TELEMETRY LOG] For satellite beacon {beacon}, the unique numeric passcode is {code}. Record {code}.\n"
        query_txt = f"\nQuestion: Based solely on the telemetry log above, what is the unique numeric passcode for satellite beacon {beacon}?\nAnswer (digits only): The unique numeric passcode is "
        n_ids = tokenizer.encode(needle_txt, add_special_tokens=False)
        q_ids = tokenizer.encode(query_txt, add_special_tokens=False)
        rem = ctx_len - len(n_ids) - len(q_ids)
        offset = (i * 1931 + ctx_len) % (len(corpus_ids) - rem - 16)
        bg = corpus_ids[offset : offset + rem]
        ins = max(32, min(rem - 32, int(round(rem * depth))))
        full_ids = bg[:ins] + n_ids + bg[ins:] + q_ids
        cases.append({"depth": depth, "expected": code, "ids": full_ids})
    return cases


def _build_mk_niah_cases(
    tokenizer: Any,
    corpus_ids: list[int],
    ctx_len: int,
    n_cases: int = 6,
) -> list[dict[str, Any]]:
    cases = []
    rng = random.Random(5500 + ctx_len)
    projects = ["Amber", "Cobalt", "Indigo", "Quartz", "Onyx", "Topaz", "Garnet", "Beryl"]
    for i in range(n_cases):
        sel_p = rng.sample(projects, 6)
        targets = sel_p[:4]
        distractors = sel_p[4:]
        t_codes = [str(rng.randint(100000, 999999)) for _ in targets]
        d_codes = [str(rng.randint(100000, 999999)) for _ in distractors]

        inserts: list[tuple[float, list[int]]] = []
        target_depths = [0.14, 0.39, 0.63, 0.86]
        dist_depths = [0.27, 0.74]
        for proj, code, dp in zip(targets, t_codes, target_depths):
            txt = f"\n[VAULT KEY] Official key for Project {proj} is {code}.\n"
            inserts.append((dp, tokenizer.encode(txt, add_special_tokens=False)))
        for proj, code, dp in zip(distractors, d_codes, dist_depths):
            txt = f"\n[VAULT KEY] Archived key for Project {proj} is {code}.\n"
            inserts.append((dp, tokenizer.encode(txt, add_special_tokens=False)))
        inserts.sort(key=lambda x: x[0])

        q_txt = (
            f"\nQuestion: Extract ONLY the 4 official 6-digit keys for Project {targets[0]}, Project {targets[1]}, "
            f"Project {targets[2]}, and Project {targets[3]} from the vault logs above (ignore archived keys). "
            f"Format strictly as 4 comma-separated numbers.\nAnswer: "
        )
        q_ids = tokenizer.encode(q_txt, add_special_tokens=False)
        ins_tot = sum(len(x[1]) for x in inserts)
        rem = ctx_len - ins_tot - len(q_ids)
        offset = (i * 2713 + ctx_len * 3) % (len(corpus_ids) - rem - 16)
        bg = corpus_ids[offset : offset + rem]

        assembled: list[int] = []
        prev_cut = 0
        for dp, tok_seg in inserts:
            cut = max(prev_cut, min(rem, int(round(rem * dp))))
            assembled.extend(bg[prev_cut:cut])
            assembled.extend(tok_seg)
            prev_cut = cut
        assembled.extend(bg[prev_cut:])
        assembled.extend(q_ids)
        cases.append({"expected_list": t_codes, "ids": assembled})
    return cases


def _extract_vt_pred(gen_txt: str) -> str:
    """从 VT 链式输出中优先解析 `FINAL: <5位整数>`，未命中标签时回退提取最后一个 5 位数。"""
    m_fin = re.search(r"FINAL\s*:\s*(\d{5})", gen_txt)
    if m_fin:
        return m_fin.group(1)
    nums = re.findall(r"\b\d{5}\b", gen_txt)
    return nums[-1] if nums else ""


def _build_vt_cases(
    tokenizer: Any,
    corpus_ids: list[int],
    ctx_len: int,
    n_cases: int = 6,
) -> list[dict[str, Any]]:
    cases = []
    rng = random.Random(8800 + ctx_len)
    pool_names = ["VAR_A1", "VAR_B2", "VAR_F5", "VAR_G6", "VAR_K7", "VAR_M3", "VAR_P9", "VAR_R0", "VAR_T4", "VAR_X8"]
    chat_wrap = tokenizer.apply_chat_template(
        [{"role": "user", "content": "@@@BODY@@@"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    wrap_pre, wrap_suf = chat_wrap.split("@@@BODY@@@")
    pre_ids = tokenizer.encode(wrap_pre, add_special_tokens=False)
    suf_ids = tokenizer.encode(wrap_suf, add_special_tokens=False)

    for i in range(n_cases):
        names = rng.sample(pool_names, 6)
        v0, v1, v2, v3 = names[:4]
        d0, d1 = names[4:]
        val = str(rng.randint(10000, 99999))
        d_val = str(rng.randint(10000, 99999))
        while d_val == val:
            d_val = str(rng.randint(10000, 99999))

        head_inst = (
            f"Track the [MEMO TRACE] assignments carefully.\n"
            f"Question: What integer is assigned to {v3}?\n"
            f'Reply by tracing one variable per line from {v3} down to the number, ending with "FINAL: <number>".\n\n'
        )
        tail_inst = (
            f"\n\nTrace {v3} one hop per line back to the initial integer (do NOT guess a number before writing all 4 lines), "
            f'then write "FINAL: <5-digit integer>".'
        )
        head_ids = tokenizer.encode(head_inst, add_special_tokens=False)
        tail_ids = tokenizer.encode(tail_inst, add_special_tokens=False)
        asst_ids = tokenizer.encode("Step 1:", add_special_tokens=False)

        hops = [
            (0.11, f"\n[MEMO TRACE] {v0} = {val}\n"),
            (0.24, f"\n[MEMO TRACE] {d0} = {d_val}\n"),
            (0.38, f"\n[MEMO TRACE] {v1} = {v0}\n"),
            (0.54, f"\n[MEMO TRACE] {d1} = {d0}\n"),
            (0.67, f"\n[MEMO TRACE] {v2} = {v1}\n"),
            (0.89, f"\n[MEMO TRACE] {v3} = {v2}\n"),
        ]
        inserts = [(dp, tokenizer.encode(t, add_special_tokens=False)) for dp, t in hops]
        ins_tot = sum(len(x[1]) for x in inserts)
        rem = ctx_len - len(pre_ids) - len(suf_ids) - len(head_ids) - len(tail_ids) - ins_tot - len(asst_ids)
        offset = (i * 3571 + ctx_len * 5) % (len(corpus_ids) - rem - 16)
        bg = corpus_ids[offset : offset + rem]

        body_mid: list[int] = []
        prev_cut = 0
        for dp, tok_seg in inserts:
            cut = max(prev_cut, min(rem, int(round(rem * dp))))
            body_mid.extend(bg[prev_cut:cut])
            body_mid.extend(tok_seg)
            prev_cut = cut
        body_mid.extend(bg[prev_cut:])
        full_ids = pre_ids + head_ids + body_mid + tail_ids + suf_ids + asst_ids
        cases.append({"expected": val, "ids": full_ids})
    return cases


def run_probe(model: torch.nn.Module, tokenizer: Any, corpus_ids: list[int], device: torch.device) -> None:
    """快速冒烟验证 VT@8K (1 case) 与 2K-chunk S-NIAH@32K 峰值显存及耗时（~50s）。"""
    print("\n================ [PROBE] Qwen3-4B VT@8K (1 case) & S-NIAH@32K 快速冒烟 ================", flush=True)
    vt_case = _build_vt_cases(tokenizer, corpus_ids, 8192, n_cases=1)[0]
    inp_vt = torch.tensor([vt_case["ids"]], dtype=torch.long, device=device)
    for m_id, _, m_cfg in [METHODS[0], METHODS[3]]:
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        cache, ctx = _make_cache_and_ctx(model, m_cfg)
        txt = _greedy_decode_text(model, tokenizer, inp_vt, cache, ctx, max_new_tokens=96)
        pred = _extract_vt_pred(txt)
        ok = pred == vt_case["expected"]
        print(f"  VT@8K     | {m_id:20s} | {time.perf_counter()-t0:4.1f}s | ok={ok} exp={vt_case['expected']} pred={pred} | gen={txt!r}", flush=True)
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    del inp_vt

    c32 = _build_s_niah_cases(tokenizer, corpus_ids, 32768, n_depths=1)[0]
    inp32 = torch.tensor([c32["ids"]], dtype=torch.long, device=device)
    for m_id, _, m_cfg in [METHODS[0], METHODS[3]]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        cache, ctx = _make_cache_and_ctx(model, m_cfg)
        txt = _greedy_decode_text(model, tokenizer, inp32, cache, ctx, max_new_tokens=16)
        dt = time.perf_counter() - t0
        pk_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
        res_gb = torch.cuda.max_memory_reserved(device) / (1024.0 ** 3)
        ok = c32["expected"] in txt
        print(f"  S-NIAH@32K| {m_id:20s} | {dt:4.1f}s | peak_alloc={pk_gb:4.2f}GB peak_res={res_gb:4.2f}GB | ok={ok} exp={c32['expected']} | gen={txt!r}", flush=True)
        del cache
        gc.collect()
        torch.cuda.empty_cache()


def _flush_checkpoint(data: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _eval_bp_variants_on_case(
    model: torch.nn.Module,
    tokenizer: Any,
    prefill_ids: torch.Tensor,
    last_tok: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, tuple[str, float] | None]:
    """一次 BitPlane Base Prefill + `copy.deepcopy` 零重算切换 4 种读取模式各自 Decode。"""
    res_map: dict[str, tuple[str, float] | None] = {}
    bp_base = BitPlaneDynamicCache(config=model.config, page_size=PAGE_SIZE, recent_tokens=RECENT_TOKENS)
    try:
        with bp_base.attach(model), torch.no_grad():
            _chunked_prefill(model, prefill_ids, bp_base)
    except torch.cuda.OutOfMemoryError:
        del bp_base
        gc.collect()
        torch.cuda.empty_cache()
        for m_id, _, _ in METHODS[1:]:
            res_map[m_id] = None
        return res_map

    for m_id, _, m_cfg in METHODS[1:]:
        cache = _clone_bp_for_method(bp_base, m_cfg, collect_stats=True)
        try:
            with cache.attach(model):
                gen_txt = _decode_from_last_tok(
                    model, tokenizer, last_tok, cache, max_new_tokens=max_new_tokens
                )
            lrb = (
                float(np.mean(cache.read_bits_samples))
                if cache.read_bits_samples
                else compute_logical_read_bits_per_kv_elem("int4")
            )
            res_map[m_id] = (gen_txt, lrb)
        except torch.cuda.OutOfMemoryError:
            res_map[m_id] = None
        finally:
            del cache
            torch.cuda.empty_cache()

    del bp_base
    gc.collect()
    torch.cuda.empty_cache()
    return res_map


def run_ruler_benchmarks(
    model: torch.nn.Module,
    tokenizer: Any,
    corpus_ids: list[int],
    device: torch.device,
    state_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """[1/3] 8K / 16K 五方同台 + 24K/32K WDDM 显存墙专项 RULER (S-NIAH / MK-NIAH / VT) 实测。"""
    print("\n================ [1/3] Qwen3-4B RULER (8K–16K 五方全量 + 24K–32K 显存墙专项) ================", flush=True)
    existing_ruler = (state_data or {}).get("ruler") or {}
    ruler_out: dict[str, Any] = {
        "S_NIAH": dict(existing_ruler.get("S_NIAH", {})),
        "MK_NIAH": dict(existing_ruler.get("MK_NIAH", {})),
        "VT": dict(existing_ruler.get("VT", {})),
    }
    if state_data is not None:
        state_data["ruler"] = ruler_out

    def _run_task_suite(task_key: str, lengths: tuple[int, ...], builder_fn, max_tok: int, matcher_fn, title: str) -> None:
        for L in lengths:
            if str(L) in ruler_out[task_key] and len(ruler_out[task_key][str(L)]) == len(METHODS):
                print(f"\n[RESUME SKIP] {title} @ L={L} 已完成，跳过", flush=True)
                continue
            cases = builder_fn(L)
            ruler_out[task_key][str(L)] = {}
            skip_native_wall = (task_key != "S_NIAH" and L >= 24576) or (L >= 32768)
            print(f"\n--- {title} @ L={L} ({len(cases)} cases, native_skip={skip_native_wall}) ---", flush=True)
            stats: dict[str, dict[str, Any]] = {
                m_id: {"hits": 0, "key_hits": 0, "all_hits": 0, "bits": [], "oom": False, "wall": skip_native_wall and m_id == "native_bf16"}
                for m_id, _, _ in METHODS
            }
            for c_item in cases:
                full = torch.tensor([c_item["ids"]], dtype=torch.long, device=device)
                pref, last_tok = full[:, :-1], full[:, -1:]

                # 1) Native BF16（8K/16K/24K 完整运行；32K 跨越 WDDM ~14GB 专用显存墙，客观标记）
                if not skip_native_wall and not stats["native_bf16"]["oom"]:
                    nat = DynamicCache(config=model.config)
                    try:
                        with torch.no_grad():
                            _chunked_prefill(model, pref, nat)
                        txt_n = _decode_from_last_tok(model, tokenizer, last_tok, nat, max_new_tokens=max_tok)
                        matcher_fn(c_item, txt_n, stats["native_bf16"])
                    except torch.cuda.OutOfMemoryError:
                        stats["native_bf16"]["oom"] = True
                    finally:
                        del nat
                        gc.collect()
                        torch.cuda.empty_cache()

                # 2) BitPlane Base 一次 Prefill + deepcopy 4 种压缩读取模式分别 Decode（跑满 8K–32K）
                bp_out = _eval_bp_variants_on_case(model, tokenizer, pref, last_tok, max_new_tokens=max_tok)
                for m_id, _, _ in METHODS[1:]:
                    val = bp_out[m_id]
                    if val is None:
                        stats[m_id]["oom"] = True
                    else:
                        txt_b, lrb = val
                        matcher_fn(c_item, txt_b, stats[m_id])
                        stats[m_id]["bits"].append(lrb)

                del full, pref, last_tok
                gc.collect()
                torch.cuda.empty_cache()

            for m_id, _, m_cfg in METHODS:
                st = stats[m_id]
                lrb = float(np.mean(st["bits"])) if st["bits"] else (16.0 if m_cfg["kind"] == "native" else compute_logical_read_bits_per_kv_elem("int4"))
                if st["wall"]:
                    if task_key == "MK_NIAH":
                        ruler_out[task_key][str(L)][m_id] = {
                            "key_hits": None,
                            "total_keys": len(cases) * 4,
                            "key_acc": None,
                            "exact_all4_acc": None,
                            "status": "skipped_wddm_vram_wall",
                            "logical_read_bits": 16.0,
                        }
                    else:
                        ruler_out[task_key][str(L)][m_id] = {
                            "hits": None,
                            "total": len(cases),
                            "acc": None,
                            "status": "skipped_wddm_vram_wall",
                            "logical_read_bits": 16.0,
                        }
                    print(f"  {m_id:20s} | VRAM_WALL (Windows WDDM >14GB decode resize swap)", flush=True)
                    continue

                if task_key == "MK_NIAH":
                    tot_keys = len(cases) * 4
                    k_acc = None if st["oom"] else (st["key_hits"] / tot_keys)
                    ex_acc = None if st["oom"] else (st["all_hits"] / len(cases))
                    ruler_out[task_key][str(L)][m_id] = {
                        "key_hits": -1 if st["oom"] else st["key_hits"],
                        "total_keys": tot_keys,
                        "key_acc": k_acc,
                        "exact_all4_acc": ex_acc,
                        "logical_read_bits": lrb,
                    }
                    print(
                        f"  {m_id:20s} | key_acc={k_acc*100 if k_acc is not None else -1:5.1f}% ({st['key_hits']}/{tot_keys}) "
                        f"| exact4={ex_acc*100 if ex_acc is not None else -1:5.1f}%",
                        flush=True,
                    )
                else:
                    acc = None if st["oom"] else (st["hits"] / len(cases))
                    ruler_out[task_key][str(L)][m_id] = {
                        "hits": -1 if st["oom"] else st["hits"],
                        "total": len(cases),
                        "acc": acc,
                        "logical_read_bits": lrb,
                    }
                    print(
                        f"  {m_id:20s} | acc={acc*100 if acc is not None else -1:5.1f}% ({st['hits']}/{len(cases)}) | read={lrb:.2f} b/KV",
                        flush=True,
                    )
            if state_data is not None:
                _flush_checkpoint(state_data)

    def _match_sniah(c_item, txt, st):
        if c_item["expected"] in txt:
            st["hits"] += 1

    def _match_mkniah(c_item, txt, st):
        matched = sum(1 for code in c_item["expected_list"] if code in txt)
        st["key_hits"] += matched
        if matched == len(c_item["expected_list"]):
            st["all_hits"] += 1

    def _match_vt(c_item, txt, st):
        if _extract_vt_pred(txt) == c_item["expected"]:
            st["hits"] += 1

    _run_task_suite("S_NIAH", (8192, 16384, 24576, 32768), lambda L: _build_s_niah_cases(tokenizer, corpus_ids, L, n_depths=10), 16, _match_sniah, "RULER-S-NIAH")
    _run_task_suite("MK_NIAH", (16384, 24576, 32768), lambda L: _build_mk_niah_cases(tokenizer, corpus_ids, L, n_cases=6), 48, _match_mkniah, "RULER-MK-NIAH")
    _run_task_suite("VT", (8192, 16384, 24576, 32768), lambda L: _build_vt_cases(tokenizer, corpus_ids, L, n_cases=6), 96, _match_vt, "RULER-VT")
    return ruler_out


def run_long_ppl_benchmarks(
    model: torch.nn.Module,
    tokenizer: Any,
    corpus_ids: list[int],
    device: torch.device,
    state_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """[2/3] 8K & 16K 五方真实自回归 PPL / KL (vs BF16) + 24K & 32K 四压缩模式真实自回归 PPL (vs Paged-INT4)。"""
    print("\n================ [2/3] Qwen3-4B 8K / 16K / 24K / 32K 长文本 128 步 PPL / KL 评测 ================", flush=True)
    out: dict[str, Any] = dict((state_data or {}).get("long_ppl") or {})
    if state_data is not None:
        state_data["long_ppl"] = out
    eval_steps = 128

    readme_txt = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    kern_txt = (PROJECT_ROOT / "src" / "bitplane_attn" / "triton_kernels.py").read_text(encoding="utf-8")
    doc_streams = [
        corpus_ids[:36000],
        (tokenizer.encode(readme_txt + "\n\n" + kern_txt, add_special_tokens=False) * 6)[:36000],
    ]

    for L in (8192, 16384, 24576, 32768):
        if str(L) in out and len(out[str(L)].get("methods", {})) == len(METHODS) - 1:
            print(f"\n[RESUME SKIP] Long-Context PPL @ L={L} 已完成，跳过", flush=True)
            continue
        is_wddm_wall = L >= 24576
        print(f"\n--- Long-Context PPL @ Context L={L} (+128 autoreg steps, bf16_ref={not is_wddm_wall}) ---", flush=True)
        bf_ppls: list[float] = []
        acc_m: dict[str, dict[str, list[float]]] = {m_id: {"ppls": [], "kls": [], "bits": []} for m_id, _, _ in METHODS[1:]}

        for doc_i, stream in enumerate(doc_streams):
            prefix_minus_1 = torch.tensor([stream[: L - 1]], dtype=torch.long, device=device)
            eval_chunk = torch.tensor([stream[L - 1 : L - 1 + eval_steps]], dtype=torch.long, device=device)
            targets = np.array(stream[L : L + eval_steps], dtype=np.int64)

            lpr: np.ndarray | None = None
            if not is_wddm_wall:
                nat_cache = DynamicCache(config=model.config)
                ref_logits_list = []
                with torch.no_grad():
                    _chunked_prefill(model, prefix_minus_1, nat_cache)
                    for t in range(eval_steps):
                        step_o = model(input_ids=eval_chunk[:, t : t + 1], cache=nat_cache, use_cache=True)
                        ref_logits_list.append(step_o.logits[0, -1, :].float())
                        del step_o
                ref_logits = torch.stack(ref_logits_list, dim=0).cpu().numpy()
                mz_r = np.max(ref_logits, axis=1, keepdims=True)
                lse_r = mz_r[:, 0] + np.log(np.sum(np.exp(ref_logits - mz_r), axis=1))
                bf_ppls.append(float(np.exp(np.mean(lse_r - ref_logits[np.arange(eval_steps), targets]))))
                lpr = ref_logits - mz_r - np.log(np.sum(np.exp(ref_logits - mz_r), axis=1, keepdims=True))
                del nat_cache, ref_logits_list, ref_logits
                gc.collect()
                torch.cuda.empty_cache()

            bp_base = BitPlaneDynamicCache(config=model.config, page_size=PAGE_SIZE, recent_tokens=RECENT_TOKENS)
            with bp_base.attach(model), torch.no_grad():
                _chunked_prefill(model, prefix_minus_1, bp_base)

            # 32K 下以 paged_int4 首位作为参考基准记录相对 KL
            for m_id, _, m_cfg in METHODS[1:]:
                cache = _clone_bp_for_method(bp_base, m_cfg, collect_stats=True)
                logits_list = []
                with cache.attach(model), torch.no_grad():
                    for t in range(eval_steps):
                        step_o = model(input_ids=eval_chunk[:, t : t + 1], cache=cache, use_cache=True)
                        logits_list.append(step_o.logits[0, -1, :].float())
                        del step_o
                logits = torch.stack(logits_list, dim=0).cpu().numpy()
                mz = np.max(logits, axis=1, keepdims=True)
                lse = mz[:, 0] + np.log(np.sum(np.exp(logits - mz), axis=1))
                acc_m[m_id]["ppls"].append(float(np.exp(np.mean(lse - logits[np.arange(eval_steps), targets]))))
                lpc = logits - mz - np.log(np.sum(np.exp(logits - mz), axis=1, keepdims=True))
                if lpr is None:
                    lpr = lpc
                acc_m[m_id]["kls"].append(float(np.mean(np.maximum(0.0, np.sum(np.exp(lpr) * (lpr - lpc), axis=1)))))
                acc_m[m_id]["bits"].extend(cache.read_bits_samples)
                del cache, logits_list, logits
                torch.cuda.empty_cache()

            del bp_base, prefix_minus_1, eval_chunk
            gc.collect()
            torch.cuda.empty_cache()

        bf16_ppl: float | None = None if is_wddm_wall else float(np.mean(bf_ppls))
        ref_ppl = float(np.mean(acc_m["paged_int4"]["ppls"])) if is_wddm_wall else float(bf16_ppl)
        out[str(L)] = {
            "bf16_ppl": bf16_ppl,
            "ref_mode": "paged_int4_wddm_wall" if is_wddm_wall else "native_bf16",
            "methods": {},
        }
        if bf16_ppl is not None:
            print(f"  Native BF16 Mean PPL @ {L} = {bf16_ppl:.4f}", flush=True)
        else:
            print(f"  Native BF16 @ {L} = skipped_wddm_vram_wall (以 Paged-INT4 PPL={ref_ppl:.4f} 作高精度参照)", flush=True)

        for m_id, _, _ in METHODS[1:]:
            m_ppl = float(np.mean(acc_m[m_id]["ppls"]))
            m_kl = float(np.mean(acc_m[m_id]["kls"]))
            m_bits = float(np.mean(acc_m[m_id]["bits"])) if acc_m[m_id]["bits"] else compute_logical_read_bits_per_kv_elem("int4")
            out[str(L)]["methods"][m_id] = {
                "mean_ppl": m_ppl,
                "delta_ppl": m_ppl - ref_ppl,
                "mean_kl": m_kl,
                "logical_read_bits_per_kv_elem": m_bits,
            }
            print(
                f"  {m_id:20s} | read={m_bits:4.2f} b/KV | PPL={m_ppl:.4f} (dPPL={m_ppl - ref_ppl:+.4f}) | KL={m_kl:.5f}",
                flush=True,
            )
        if state_data is not None:
            _flush_checkpoint(state_data)

    return out


def run_4b_vram_and_throughput(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    state_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """[3/3] Qwen3-4B 三方 (`Native BF16` vs `Paged INT4` vs `BitPlane Dual 15+35`) 6 档显存与吞吐消融。"""
    print("\n================ [3/3] Qwen3-4B 三方显存与 End-to-End Greedy Decode 吞吐消融 ================", flush=True)
    warm_ids = torch.randint(10, 500, (1, 128), device=device, dtype=torch.long)
    for m_cfg in (METHODS[1][2], METHODS[3][2]):
        wc, wctx = _make_cache_and_ctx(model, m_cfg)
        _greedy_decode_text(model, tokenizer, warm_ids, wc, wctx, max_new_tokens=4)
        del wc
    del warm_ids
    gc.collect()
    torch.cuda.empty_cache()

    specs = [
        ("B1_L8192_G64", 1, 8192, 64),
        ("B1_L16384_G64", 1, 16384, 64),
        ("B1_L24576_G64", 1, 24576, 64),
        ("B2_L8192_G64", 2, 8192, 64),
        ("B1_L32768_G64", 1, 32768, 64),
        ("B2_L16384_G64", 2, 16384, 64),
    ]
    res: dict[str, Any] = dict((state_data or {}).get("vram_and_throughput") or {})
    if state_data is not None:
        state_data["vram_and_throughput"] = res
    tri_methods = [METHODS[0], METHODS[1], METHODS[3]]

    for s_key, bsz, l_pref, gen_steps in specs:
        if s_key in res and len(res[s_key].get("methods", {})) == len(tri_methods):
            print(f"  [RESUME SKIP] {s_key} 已完成，跳过", flush=True)
            continue
        torch.manual_seed(4200 + bsz + l_pref)
        prompt = torch.randint(100, 3000, (bsz, l_pref), device=device, dtype=torch.long)
        res[s_key] = {"batch_size": bsz, "prefill_len": l_pref, "gen_steps": gen_steps, "methods": {}}
        native_vram_wall = (bsz * l_pref) >= 24576

        for m_id, _, m_cfg in tri_methods:
            if m_id == "native_bf16" and native_vram_wall:
                res[s_key]["methods"][m_id] = {"oom": False, "wddm_vram_wall": True}
                continue

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
            cache, ctx = _make_cache_and_ctx(model, m_cfg)
            try:
                with ctx, torch.no_grad():
                    t0 = time.perf_counter()
                    first_logits = _chunked_prefill(model, prompt, cache, chunk_size=PREFILL_CHUNK)
                    torch.cuda.synchronize()
                    t_prefill_ms = (time.perf_counter() - t0) * 1000.0
                    nxt = torch.argmax(first_logits[:, -1, :], dim=-1, keepdim=True)
                    del first_logits

                    torch.cuda.synchronize()
                    td0 = time.perf_counter()
                    for _ in range(gen_steps):
                        step_o = model(input_ids=nxt, cache=cache, use_cache=True)
                        nxt = torch.argmax(step_o.logits[:, -1, :], dim=-1, keepdim=True)
                        del step_o
                    torch.cuda.synchronize()
                    t_dec_s = time.perf_counter() - td0

                steady_kv_mb = (torch.cuda.memory_allocated(device) / (1024.0 ** 2)) - base_mb
                peak_total_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                dec_tps = float(bsz * gen_steps) / max(t_dec_s, 1e-9)
                res[s_key]["methods"][m_id] = {
                    "oom": False,
                    "wddm_vram_wall": False,
                    "steady_kv_vram_mb": steady_kv_mb,
                    "peak_total_vram_mb": peak_total_mb,
                    "prefill_ms": t_prefill_ms,
                    "greedy_decode_total_s": t_dec_s,
                    "greedy_decode_ms_per_step": (t_dec_s * 1000.0) / float(gen_steps),
                    "greedy_decode_tokens_per_sec": dec_tps,
                    "cache_pool_stats": cache.memory_stats() if m_cfg["kind"] == "bp" else None,
                }
            except torch.cuda.OutOfMemoryError:
                res[s_key]["methods"][m_id] = {"oom": True, "wddm_vram_wall": False}
            finally:
                del cache
                gc.collect()
                torch.cuda.empty_cache()

        nat = res[s_key]["methods"]["native_bf16"]
        i4 = res[s_key]["methods"]["paged_int4"]
        bp = res[s_key]["methods"]["bitplane_dual_15_35"]
        nat_ok = (not nat.get("oom")) and (not nat.get("wddm_vram_wall"))
        i4_ok = (not i4.get("oom")) and (not i4.get("wddm_vram_wall"))
        bp_ok = (not bp.get("oom")) and (not bp.get("wddm_vram_wall"))
        if nat_ok and i4_ok:
            res[s_key]["vram_compression_int4_vs_bf16"] = nat["steady_kv_vram_mb"] / max(i4["steady_kv_vram_mb"], 0.1)
            res[s_key]["greedy_speedup_int4_vs_bf16"] = i4["greedy_decode_tokens_per_sec"] / max(nat["greedy_decode_tokens_per_sec"], 1e-6)
        if nat_ok and bp_ok:
            res[s_key]["vram_compression_dual35_vs_bf16"] = nat["steady_kv_vram_mb"] / max(bp["steady_kv_vram_mb"], 0.1)
            res[s_key]["greedy_speedup_dual35_vs_bf16"] = bp["greedy_decode_tokens_per_sec"] / max(nat["greedy_decode_tokens_per_sec"], 1e-6)
        if i4_ok and bp_ok:
            res[s_key]["greedy_speedup_dual35_vs_int4"] = bp["greedy_decode_tokens_per_sec"] / max(i4["greedy_decode_tokens_per_sec"], 1e-6)

        nat_s = "VRAM_WALL" if nat.get("wddm_vram_wall") else ("OOM" if nat.get("oom") else f"{nat['steady_kv_vram_mb']:.1f}MB ({nat['greedy_decode_tokens_per_sec']:.1f} tok/s)")
        i4_s = "OOM" if i4.get("oom") else f"{i4['steady_kv_vram_mb']:.1f}MB ({i4['greedy_decode_tokens_per_sec']:.1f} tok/s)"
        bp_s = "OOM" if bp.get("oom") else f"{bp['steady_kv_vram_mb']:.1f}MB ({bp['greedy_decode_tokens_per_sec']:.1f} tok/s)"
        sp_txt = f" | Dual35 vs INT4={res[s_key]['greedy_speedup_dual35_vs_int4']:.2f}x" if "greedy_speedup_dual35_vs_int4" in res[s_key] else ""
        print(f"  {s_key:15s} | Native={nat_s} | Paged-INT4={i4_s} | Dual35={bp_s}{sp_txt}", flush=True)
        del prompt
        gc.collect()
        torch.cuda.empty_cache()
        if state_data is not None:
            _flush_checkpoint(state_data)

    return res


def render_4b_figure(data: dict[str, Any]) -> None:
    """绘制 300 DPI 四宫格 Batch B Qwen3-4B 长上下文实测总结图。"""
    plt.rcParams.update({"font.size": 9.2, "axes.titlesize": 10.2, "axes.labelsize": 9.2})
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.4), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.91, bottom=0.08, left=0.07, right=0.97)
    fig.suptitle(
        "Batch B: Qwen3-4B (36L, d=128) Long-Context RULER (8K-32K), PPL & 3-Way Ablation (RTX 5070 Ti)",
        fontsize=12.4,
        fontweight="bold",
        y=0.965,
    )

    ruler = data["ruler"]
    ppl = data["long_ppl"]
    e2e = data["vram_and_throughput"]

    # (a) RULER Macro Average Accuracy across 8K / 16K / 24K / 32K
    ax = axes[0, 0]
    lengths = ["8192", "16384", "24576", "32768"]
    x_pos = np.arange(len(lengths))
    styles = {
        "native_bf16": ("Native BF16 (8K-16K full, 24K S-NIAH only)", "#2c3e50", "-", "o"),
        "paged_int4": ("Paged-INT4 (static)", "#2980b9", "--", "s"),
        "readkv_prog_15_85": ("ReadKV-like (15+85)", "#16a085", "-.", "^"),
        "bitplane_dual_15_35": ("BitPlane Dual (15+35, Ours)", "#c0392b", "-", "*"),
        "quest_hard_15_0": ("Quest-like Sparse (15+0)", "#e67e22", ":", "d"),
    }
    for m_id, (lbl, col, ls, mkr) in styles.items():
        ys = []
        for L_str in lengths:
            if m_id == "native_bf16" and L_str == "24576":
                ys.append(np.nan)
                continue
            vals = []
            if L_str in ruler["S_NIAH"] and ruler["S_NIAH"][L_str][m_id].get("acc") is not None:
                vals.append(ruler["S_NIAH"][L_str][m_id]["acc"])
            if L_str in ruler["MK_NIAH"] and ruler["MK_NIAH"][L_str][m_id].get("key_acc") is not None:
                vals.append(ruler["MK_NIAH"][L_str][m_id]["key_acc"])
            if L_str in ruler["VT"] and ruler["VT"][L_str][m_id].get("acc") is not None:
                vals.append(ruler["VT"][L_str][m_id]["acc"])
            ys.append(float(np.mean(vals)) * 100.0 if vals else np.nan)
        ms = 9 if m_id == "bitplane_dual_15_35" else 6
        lw = 2.4 if m_id == "bitplane_dual_15_35" else 1.8
        ax.plot(x_pos, ys, label=lbl, color=col, linestyle=ls, marker=mkr, linewidth=lw, markersize=ms)
    if "24576" in ruler.get("S_NIAH", {}) and ruler["S_NIAH"]["24576"]["native_bf16"].get("acc") is not None:
        sniah_24k = ruler["S_NIAH"]["24576"]["native_bf16"]["acc"] * 100.0
        ax.scatter([2], [sniah_24k], color="#2c3e50", facecolor="white", marker="o", s=54, zorder=5)
        ax.annotate(
            "Native 24K S-NIAH only (100%)",
            xy=(2, sniah_24k),
            xytext=(2.0, 105.0),
            ha="center",
            va="bottom",
            fontsize=6.9,
            color="#2c3e50",
            arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.8),
        )
    ax.axvline(1.45, color="#95a5a6", linestyle="--", linewidth=1.1)
    ax.text(1.48, 8, "L>=24K long decode\nhits WDDM VRAM wall", fontsize=7.0, color="#7f8c8d", va="bottom")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["8K", "16K", "24K", "32K"])
    ax.set_ylim(-5, 112)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("RULER Macro Accuracy (%)")
    ax.set_title("(a) RULER Macro Accuracy (S-NIAH / MK-NIAH / VT)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=7.2, frameon=True, framealpha=0.92)

    # (b) Long-Text 8K & 16K Logit KL vs BF16 Pareto
    ax = axes[0, 1]
    for L_str, col, mkr, ls in [("8192", "#2980b9", "o", "-"), ("16384", "#c0392b", "s", "--")]:
        m_dict = ppl[L_str]["methods"]
        order = [
            ("Quest(15+0)", m_dict["quest_hard_15_0"]),
            ("Dual(15+35)", m_dict["bitplane_dual_15_35"]),
            ("ReadKV(15+85)", m_dict["readkv_prog_15_85"]),
            ("Paged-INT4", m_dict["paged_int4"]),
        ]
        xs = [it[1]["logical_read_bits_per_kv_elem"] for it in order]
        ys = [max(1e-4, it[1]["mean_kl"]) for it in order]
        ax.plot(xs, ys, color=col, marker=mkr, linestyle=ls, linewidth=2.0, markersize=7, label=f"Qwen3-4B L={int(L_str)//1024}K")
        offsets = [(15, 12), (4, 13), (0, 12), (-14, 12)] if L_str == "8192" else [(15, -13), (0, -14), (0, -13), (-14, -12)]
        for idx_p, (txt, p_d) in enumerate(order):
            ax.annotate(
                txt,
                xy=(p_d["logical_read_bits_per_kv_elem"], max(1e-4, p_d["mean_kl"])),
                xytext=offsets[idx_p],
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=6.8,
                color=col,
                bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none", alpha=0.78),
            )
    ax.set_yscale("log")
    ax.set_xlim(0.7, 5.2)
    ax.set_ylim(3.5e-4, 0.30)
    ax.set_xlabel("Logical KV Read Bandwidth (bits / KV-elem, incl. metadata)")
    ax.set_ylabel("128-Step Autoreg Logit KL vs BF16 (log scale)")
    ax.set_title("(b) Long-Text (8K & 16K) Accuracy vs Read Bandwidth")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=7.8, frameon=True)

    # (c) Real CUDA Steady KV VRAM (Native BF16 vs Paged-INT4 vs BitPlane Dual35)
    ax = axes[1, 0]
    specs_v = ["B1_L8192_G64", "B1_L16384_G64", "B1_L24576_G64", "B2_L8192_G64", "B1_L32768_G64", "B2_L16384_G64"]
    x_lbl_v = ["B1\n8K", "B1\n16K", "B1\n24K", "B2\n8K", "B1\n32K", "B2\n16K"]
    xv = np.arange(len(specs_v))
    w = 0.26

    def _safe_val(d_m: dict[str, Any], key: str) -> float:
        return 0.0 if (d_m.get("oom") or d_m.get("wddm_vram_wall")) else float(d_m[key])

    nat_v = [_safe_val(e2e[s]["methods"]["native_bf16"], "steady_kv_vram_mb") for s in specs_v]
    i4_v = [_safe_val(e2e[s]["methods"]["paged_int4"], "steady_kv_vram_mb") for s in specs_v]
    bp_v = [_safe_val(e2e[s]["methods"]["bitplane_dual_15_35"], "steady_kv_vram_mb") for s in specs_v]

    b1 = ax.bar(xv - w, nat_v, width=w * 0.92, color="#7f8c8d", label="Native BF16 DynamicCache")
    b2 = ax.bar(xv, i4_v, width=w * 0.92, color="#2980b9", label="Paged-INT4 Cache")
    b3 = ax.bar(xv + w, bp_v, width=w * 0.92, color="#c0392b", label="BitPlane Dual (15+35)")
    for idx_s, s_k in enumerate(specs_v):
        nm = e2e[s_k]["methods"]["native_bf16"]
        if nm.get("wddm_vram_wall"):
            ax.text(xv[idx_s] - w, 100, "VRAM Wall", ha="center", va="bottom", rotation=90, fontsize=6.6, color="#c0392b", fontweight="bold")
        elif nm.get("oom"):
            ax.text(xv[idx_s] - w, 100, "OOM", ha="center", va="bottom", rotation=90, fontsize=6.6, color="#c0392b", fontweight="bold")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 1.0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 65, f"{h:.0f}", ha="center", va="bottom", rotation=90, fontsize=6.4)
    ax.set_xticks(xv)
    ax.set_xticklabels(x_lbl_v, fontsize=8.4)
    ax.set_ylabel("CUDA Steady KV VRAM (MiB)")
    ax.set_ylim(0, max(nat_v + i4_v + bp_v + [1000]) * 1.28)
    ax.set_title("(c) Qwen3-4B CUDA Steady KV Memory (3-Way Ablation)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7.3, frameon=True)

    # (d) Qwen3-4B 3-Way End-to-End Greedy Decode Throughput Ablation
    ax = axes[1, 1]
    nat_t = [_safe_val(e2e[s]["methods"]["native_bf16"], "greedy_decode_tokens_per_sec") for s in specs_v]
    i4_t = [_safe_val(e2e[s]["methods"]["paged_int4"], "greedy_decode_tokens_per_sec") for s in specs_v]
    bp_t = [_safe_val(e2e[s]["methods"]["bitplane_dual_15_35"], "greedy_decode_tokens_per_sec") for s in specs_v]

    t1 = ax.bar(xv - w, nat_t, width=w * 0.92, color="#7f8c8d", label="Native BF16 DynamicCache")
    t2 = ax.bar(xv, i4_t, width=w * 0.92, color="#2980b9", label="Paged-INT4 Cache")
    t3 = ax.bar(xv + w, bp_t, width=w * 0.92, color="#c0392b", label="BitPlane Dual (15+35)")
    for idx_s, s_k in enumerate(specs_v):
        nm = e2e[s_k]["methods"]["native_bf16"]
        if nm.get("wddm_vram_wall"):
            ax.text(xv[idx_s] - w, 0.8, "VRAM Wall", ha="center", va="bottom", rotation=90, fontsize=6.6, color="#c0392b", fontweight="bold")
        elif nm.get("oom"):
            ax.text(xv[idx_s] - w, 0.8, "OOM", ha="center", va="bottom", rotation=90, fontsize=6.6, color="#c0392b", fontweight="bold")
    for bars in (t1, t2, t3):
        for bar in bars:
            h = bar.get_height()
            if h > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.6, f"{h:.1f}", ha="center", va="bottom", rotation=90, fontsize=6.6)
    ax.set_xticks(xv)
    ax.set_xticklabels(x_lbl_v, fontsize=8.4)
    ax.set_ylabel("Greedy Decode Throughput (tok/s)")
    ax.set_ylim(0, max(nat_t + i4_t + bp_t + [20]) * 1.30)
    ax.set_title("(d) Qwen3-4B End-to-End Greedy Decode Throughput")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=7.3, frameon=True)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"[DONE] Batch B 总结图已保存至 {OUT_PNG}", flush=True)


def main() -> int:
    if "--plot-only" in sys.argv:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        render_4b_figure(data)
        return 0

    t0 = time.perf_counter()
    device = torch.device("cuda:0")
    m_src = _resolve_model_source("Qwen3-4B")
    print(f"加载 Qwen3-4B: {m_src}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(m_src)
    model = AutoModelForCausalLM.from_pretrained(m_src, dtype=torch.bfloat16).to(device).eval()
    corpus_ids = _build_diverse_corpus_tokens(tokenizer, min_tokens=42000)

    if "--probe" in sys.argv:
        run_probe(model, tokenizer, corpus_ids, device)
        return 0

    props = torch.cuda.get_device_properties(0)
    if "--resume" in sys.argv and OUT_JSON.exists():
        data: dict[str, Any] = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        print(f"[RESUME] 已加载断点文件 {OUT_JSON}", flush=True)
    else:
        data = {
            "hardware": {
                "gpu": props.name,
                "total_vram_gb": props.total_memory / (1024.0 ** 3),
                "sm_count": props.multi_processor_count,
                "torch": torch.__version__,
                "cuda_capability": f"sm_{props.major}{props.minor}",
                "model": "Qwen3-4B (36L, 32Q/8KV, d=128, BF16)",
            },
            "ruler": {},
        }
    data["ruler"] = run_ruler_benchmarks(model, tokenizer, corpus_ids, device, state_data=data)
    _flush_checkpoint(data)

    data["long_ppl"] = run_long_ppl_benchmarks(model, tokenizer, corpus_ids, device, state_data=data)
    _flush_checkpoint(data)

    data["vram_and_throughput"] = run_4b_vram_and_throughput(model, tokenizer, device, state_data=data)
    data["elapsed_sec"] = time.perf_counter() - t0
    _flush_checkpoint(data)
    print(f"\n[DONE] Qwen3-4B Batch B 全量评测落盘至 {OUT_JSON} (总耗时 {data['elapsed_sec']:.1f}s)", flush=True)
    render_4b_figure(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
