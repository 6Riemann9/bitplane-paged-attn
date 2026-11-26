#!/usr/bin/env python3
"""
Phase 1B 步骤 1：在真实模型上先验真 Single-Pass Anchor-Gated Bit-Plane 的稀疏跳过率与精度。

在 Qwen3.5-0.8B (Hybrid 3:1) 与 Qwen3-0.6B (Dense) 的 L=2048 长文档续写 (128 步 PPL/KL)
与 5 深度 (10%–90%) 大海捞针上，扫描 (tau_hot, tau_skip) 阈值对，校验是否能找到满足以下全部门槛的统一默认配置：
  - hot_ratio <= 0.20, skip_ratio >= 0.45 (物理读取比特显著低于 INT4 的 9.25 bit)
  - 续写 KL < 0.015 且 5 深度找针双双保持 5/5 (100%)
"""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
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

from bitplane_attn.quant_bitplane import (  # noqa: E402
    QuantizedBitPlanePages,
    _build_quantized_pages,
    paged_bitplane_attention_decode,
)

OUT_JSON = PROJECT_ROOT / "results" / "gated_threshold_sweep.json"


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

PREFIX_LEN = 2048
PAGE_SIZE = 32
RECENT_TOKENS = 32
HIST_LEN = 2016
CONT_STEPS = 128
NEEDLE_DECODE_STEPS = 16

THRESHOLD_PAIRS = [
    (3.0, 6.5),
    (3.5, 7.5),
    (4.0, 8.5),
    (4.5, 9.5),
    (5.0, 10.5),
]


@dataclass
class _ProbeCtx:
    enabled: bool = False
    mode: str = "fp16"
    tau_hot: float = 4.0
    tau_skip: float = 8.5
    pages_by_layer: dict[int, QuantizedBitPlanePages] = field(default_factory=dict)
    stats: list[dict[str, float]] = field(default_factory=list)


_CTX = _ProbeCtx()


def _gated_probe_interface(
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

    if (not _CTX.enabled) or (not isinstance(module, (Qwen3Attention, Qwen3_5Attention))):
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )

    l_idx = int(module.layer_idx)
    out, st = paged_bitplane_attention_decode(
        query=query,
        key=key,
        value=value,
        scaling=scaling,
        attention_mask=attention_mask,
        mode=_CTX.mode,
        page_size=PAGE_SIZE,
        recent_tokens=RECENT_TOKENS,
        hist_len=HIST_LEN,
        tau_hot=_CTX.tau_hot,
        tau_skip=_CTX.tau_skip,
        pages_cache=_CTX.pages_by_layer.get(l_idx),
    )
    _CTX.stats.append(
        {
            "hot_ratio": float(st.get("hot_ratio", 0.0)),
            "warm_ratio": float(st.get("warm_ratio", 0.0)),
            "skip_ratio": float(st.get("skip_ratio", 0.0)),
            "phys_read_bits": float(st.get("phys_read_bits", 16.0)),
        }
    )
    return out, None


ALL_ATTENTION_FUNCTIONS.register("gated_sweep_probe", _gated_probe_interface)


def _bind(model: torch.nn.Module, impl: str) -> None:
    if hasattr(model, "config") and model.config is not None:
        model.config._attn_implementation = impl
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            model.config.text_config._attn_implementation = impl
    for mod in model.modules():
        if hasattr(mod, "config") and mod.config is not None:
            mod.config._attn_implementation = impl


def _extract_pages(pkv: Any, attn_layers: list[int]) -> dict[int, QuantizedBitPlanePages]:
    out: dict[int, QuantizedBitPlanePages] = {}
    for l_idx in attn_layers:
        if hasattr(pkv, "layers"):
            k_t, v_t = pkv.layers[l_idx].keys, pkv.layers[l_idx].values
        else:
            k_t, v_t = pkv.key_cache[l_idx], pkv.value_cache[l_idx]
        out[l_idx] = _build_quantized_pages(k_t, v_t, hist_len=HIST_LEN, page_size=PAGE_SIZE)
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
        seqs.append({
            "name": name,
            "prefix_ids": torch.tensor([chunk[:PREFIX_LEN]], dtype=torch.long, device=device),
            "cont_ids": torch.tensor([chunk[PREFIX_LEN:]], dtype=torch.long, device=device),
        })
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
        n_ids = tokenizer.encode(f" [AUDIT RECORD] The official verification code for {proj} is {code}. ", add_special_tokens=False)
        q_ids = tokenizer.encode(f" Based strictly on the audit record above, the official verification code for {proj} is", add_special_tokens=False)
        rem = PREFIX_LEN - len(n_ids) - len(q_ids)
        b_len = max(16, int(round(rem * depth)))
        a_len = rem - b_len
        b_ids = (f_ids * ((b_len // len(f_ids)) + 2))[:b_len]
        a_ids = (f_ids * ((a_len // len(f_ids)) + 2))[:a_len]
        cases.append({
            "name": name, "depth": depth, "code": code,
            "input_ids": torch.tensor([b_ids + n_ids + a_ids + q_ids], dtype=torch.long, device=device),
        })
    return cases


def _sweep_model(model_name: str, is_35: bool, device: torch.device) -> dict[str, Any]:
    print(f"\n=== 扫描 Anchor-Gated Single-Pass 阈值: {model_name} ===", flush=True)
    m_src = _resolve_model_source(model_name)
    tokenizer = AutoTokenizer.from_pretrained(m_src)
    model = (Qwen3_5ForConditionalGeneration if is_35 else AutoModelForCausalLM).from_pretrained(
        m_src, dtype=torch.bfloat16
    ).to(device).eval()
    attn_layers = sorted({int(m.layer_idx) for m in model.modules() if isinstance(m, (Qwen3Attention, Qwen3_5Attention))})
    _bind(model, "gated_sweep_probe")

    docs = _build_docs(tokenizer, device)
    needles = _build_needles(tokenizer, device)

    doc_prep = []
    for d_item in docs:
        _CTX.enabled = False
        with torch.no_grad():
            pkv = model(input_ids=d_item["prefix_ids"][:, :-1], use_cache=True).past_key_values
        pages = _extract_pages(pkv, attn_layers)
        eval_chunk = torch.cat([d_item["prefix_ids"][:, -1:], d_item["cont_ids"][:, :-1]], dim=1)
        targets = d_item["cont_ids"][0].cpu().numpy()

        # FP16 reference
        _CTX.enabled = True
        _CTX.mode = "fp16"
        _CTX.pages_by_layer = pages
        with torch.no_grad():
            out_fp = model(input_ids=eval_chunk, past_key_values=copy.deepcopy(pkv), use_cache=True)
        _CTX.enabled = False
        ref_logits = out_fp.logits[0].float().cpu().numpy()
        max_z = np.max(ref_logits, axis=1, keepdims=True)
        lse = max_z[:, 0] + np.log(np.sum(np.exp(ref_logits - max_z), axis=1))
        fp_ppl = float(np.exp(np.mean(lse - ref_logits[np.arange(targets.shape[0]), targets])))
        doc_prep.append({"pkv": pkv, "eval_chunk": eval_chunk, "targets": targets, "pages": pages, "ref_logits": ref_logits, "fp_ppl": fp_ppl})

    needle_prep = []
    for n_item in needles:
        _CTX.enabled = False
        with torch.no_grad():
            pkv_n = model(input_ids=n_item["input_ids"][:, :-1], use_cache=True).past_key_values
        needle_prep.append({
            "pkv": pkv_n, "last_tok": n_item["input_ids"][:, -1:],
            "code": n_item["code"], "pages": _extract_pages(pkv_n, attn_layers),
        })

    fp16_ppl = float(np.mean([x["fp_ppl"] for x in doc_prep]))
    results = {"fp16_ppl": fp16_ppl, "configs": {}}

    for th, ts in THRESHOLD_PAIRS:
        label = f"tau_{th:.1f}_{ts:.1f}"
        ppls, kls, hots, warms, skips, bits = [], [], [], [], [], []
        for dp in doc_prep:
            _CTX.enabled = True
            _CTX.mode = "bitplane_gated"
            _CTX.tau_hot, _CTX.tau_skip = th, ts
            _CTX.pages_by_layer = dp["pages"]
            _CTX.stats.clear()
            with torch.no_grad():
                out = model(input_ids=dp["eval_chunk"], past_key_values=copy.deepcopy(dp["pkv"]), use_cache=True)
            _CTX.enabled = False
            logits = out.logits[0].float().cpu().numpy()
            mz = np.max(logits, axis=1, keepdims=True)
            lse = mz[:, 0] + np.log(np.sum(np.exp(logits - mz), axis=1))
            ppls.append(float(np.exp(np.mean(lse - logits[np.arange(dp["targets"].shape[0]), dp["targets"]]))))
            mr = np.max(dp["ref_logits"], axis=1, keepdims=True)
            lpr = dp["ref_logits"] - mr - np.log(np.sum(np.exp(dp["ref_logits"] - mr), axis=1, keepdims=True))
            lpc = logits - mz - np.log(np.sum(np.exp(logits - mz), axis=1, keepdims=True))
            kls.append(float(np.mean(np.maximum(0.0, np.sum(np.exp(lpr) * (lpr - lpc), axis=1)))))
            hots.extend([s["hot_ratio"] for s in _CTX.stats])
            warms.extend([s["warm_ratio"] for s in _CTX.stats])
            skips.extend([s["skip_ratio"] for s in _CTX.stats])
            bits.extend([s["phys_read_bits"] for s in _CTX.stats])

        hits = 0
        for np_it in needle_prep:
            _CTX.enabled = True
            _CTX.mode = "bitplane_gated"
            _CTX.tau_hot, _CTX.tau_skip = th, ts
            _CTX.pages_by_layer = np_it["pages"]
            pkv_g = copy.deepcopy(np_it["pkv"])
            cur = np_it["last_tok"]
            g_ids = []
            with torch.no_grad():
                for _ in range(NEEDLE_DECODE_STEPS):
                    o = model(input_ids=cur, past_key_values=pkv_g, use_cache=True)
                    pkv_g = o.past_key_values
                    cur = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
                    g_ids.append(int(cur.item()))
            _CTX.enabled = False
            if np_it["code"] in tokenizer.decode(g_ids, skip_special_tokens=True):
                hits += 1

        cfg_res = {
            "tau_hot": th,
            "tau_skip": ts,
            "mean_ppl": float(np.mean(ppls)),
            "delta_ppl": float(np.mean(ppls)) - fp16_ppl,
            "mean_kl": float(np.mean(kls)),
            "needle_hits": hits,
            "needle_total": len(needle_prep),
            "hot_ratio": float(np.mean(hots)),
            "warm_ratio": float(np.mean(warms)),
            "skip_ratio": float(np.mean(skips)),
            "phys_read_bits": float(np.mean(bits)),
        }
        results["configs"][label] = cfg_res
        print(
            f"  ({th:3.1f}, {ts:4.1f}) | hot={cfg_res['hot_ratio']*100:5.1f}% warm={cfg_res['warm_ratio']*100:5.1f}% skip={cfg_res['skip_ratio']*100:5.1f}% "
            f"| read={cfg_res['phys_read_bits']:4.2f}b (vs INT4 9.25b) | PPL={cfg_res['mean_ppl']:.3f} (dPPL={cfg_res['delta_ppl']:+.3f}) "
            f"| KL={cfg_res['mean_kl']:.5f} | Needles={hits}/5",
            flush=True,
        )

    _CTX.pages_by_layer.clear()
    del model, doc_prep, needle_prep
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main() -> int:
    t0 = time.perf_counter()
    device = torch.device("cuda:0")
    res = {
        "Qwen3.5-0.8B": _sweep_model("Qwen3.5-0.8B", True, device),
        "Qwen3-0.6B": _sweep_model("Qwen3-0.6B", False, device),
        "elapsed_sec": time.perf_counter() - t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[DONE] 阈值扫描完成，耗时 {res['elapsed_sec']:.1f}s，结果保存至 {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
