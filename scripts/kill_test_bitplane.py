#!/usr/bin/env python3
"""
Bit-Plane Paged Attention Kill Test（RTX 5070 Ti GPU 真机实证）。

在最新 Hybrid 架构 Qwen/Qwen3.5-0.8B（18 层 Gated DeltaNet + 6 层 4:1 GQA 全局注意力）
与 Dense 架构 Qwen/Qwen3-0.6B（28 层 2:1 GQA 全局注意力）上，严格零因果泄密运行三大评测：
  1. 任务 1（L=2048 长文档续写 128 步 PPL 与 KL 散度）：对比 fp16 / int4 / int2 / base2 /
     quest(per-head & GQA) / bitplane(10%, 15% max/mass, 25%) / droptail / key_only / val_only；
  2. 任务 2（L=2048 五深度 10%–90% 大海捞针精确检索）：评估自回归命中率 (X/5) 与密钥 token 平均概率；
  3. 任务 3（Hybrid 稀缺全局层 vs Dense 稠密层机理诊断）：逐层分析 K/V 通道尖峰、方差集中度、
     页内 INT2 误差及 Key-only / Value-only 端到端敏感度。

输出：
  - `results/kill_test_bitplane.json`
  - `results/kill_test_bitplane.png`（2x2 300 DPI 无遮挡英文学术图表）
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import platform
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, eager_attention_forward
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForConditionalGeneration,
)

# 兼容源码直接导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bitplane_attn.quant_bitplane import (  # noqa: E402
    QuantizedBitPlanePages,
    _build_quantized_pages,
    compute_layer_outlier_stats,
    paged_bitplane_attention_decode,
)


RESULTS_DIR = PROJECT_ROOT / "results"
OUT_JSON = RESULTS_DIR / "kill_test_bitplane.json"
OUT_FIG = RESULTS_DIR / "kill_test_bitplane.png"


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
HIST_LEN = 2016  # 63 页 * 32 = 2016；[2016, L_k) 全部严格保持全精
CONT_STEPS = 128
NEEDLE_DECODE_STEPS = 16


@dataclass
class MethodSpec:
    """描述一种 Kill Test 评测配置。"""

    name: str
    mode: str
    top_ratio: float = 0.15
    routing: str = "page_max"
    gqa_shared: bool = True


PPL_METHODS: list[MethodSpec] = [
    MethodSpec("fp16", "fp16"),
    MethodSpec("int4", "int4"),
    MethodSpec("int2", "int2"),
    MethodSpec("base2", "base2"),
    MethodSpec("quest_15_perhead", "quest", top_ratio=0.15, gqa_shared=False),
    MethodSpec("quest_15_gqa", "quest", top_ratio=0.15, gqa_shared=True),
    MethodSpec("quest_10_gqa", "quest", top_ratio=0.10, gqa_shared=True),
    MethodSpec("bitplane_15_max", "bitplane", top_ratio=0.15, routing="page_max", gqa_shared=True),
    MethodSpec("bitplane_15_mass", "bitplane", top_ratio=0.15, routing="page_mass", gqa_shared=True),
    MethodSpec("bitplane_10_max", "bitplane", top_ratio=0.10, routing="page_max", gqa_shared=True),
    MethodSpec("bitplane_25_max", "bitplane", top_ratio=0.25, routing="page_max", gqa_shared=True),
    MethodSpec("bitplane_15_droptail", "bitplane_droptail", top_ratio=0.15, routing="page_max", gqa_shared=True),
    MethodSpec("key_only_int2", "key_only_int2"),
    MethodSpec("val_only_int2", "val_only_int2"),
]

NEEDLE_METHODS: list[MethodSpec] = [
    MethodSpec("fp16", "fp16"),
    MethodSpec("int4", "int4"),
    MethodSpec("int2", "int2"),
    MethodSpec("base2", "base2"),
    MethodSpec("quest_15_perhead", "quest", top_ratio=0.15, gqa_shared=False),
    MethodSpec("quest_15_gqa", "quest", top_ratio=0.15, gqa_shared=True),
    MethodSpec("bitplane_15_max", "bitplane", top_ratio=0.15, routing="page_max", gqa_shared=True),
    MethodSpec("bitplane_15_mass", "bitplane", top_ratio=0.15, routing="page_mass", gqa_shared=True),
    MethodSpec("bitplane_10_max", "bitplane", top_ratio=0.10, routing="page_max", gqa_shared=True),
    MethodSpec("bitplane_15_droptail", "bitplane_droptail", top_ratio=0.15, routing="page_max", gqa_shared=True),
]


@dataclass
class BitPlaneHookState:
    """管理全局注意力层拦截、预计算页缓存及每步运行时统计的共享上下文。"""

    enabled: bool = False
    spec: MethodSpec = field(default_factory=lambda: MethodSpec("fp16", "fp16"))
    hist_len: int = HIST_LEN
    pages_by_layer: dict[int, QuantizedBitPlanePages] = field(default_factory=dict)
    step_stats: list[dict[str, float]] = field(default_factory=list)

    def reset_stats(self) -> None:
        self.step_stats.clear()


_HOOK_STATE = BitPlaneHookState()


def _bitplane_attention_interface(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    兼容 Qwen3 与 Qwen3.5 全部全局 Softmax 注意力层的统一拦截接口。
    严格保障多步查询 (L_q > 1) 的显式因果掩码与零未来泄密。
    """
    # 若上游未传递显式 mask 且存在多步查询，补充严格下三角因果掩码防止未来泄密
    if attention_mask is None and query.shape[2] > 1:
        lq, lk = query.shape[2], key.shape[2]
        past_len = lk - lq
        q_pos = torch.arange(past_len, lk, device=query.device)[:, None]
        k_pos = torch.arange(lk, device=query.device)[None, :]
        causal = torch.where(k_pos <= q_pos, 0.0, torch.finfo(query.dtype).min).to(query.dtype)
        attention_mask = causal[None, None, :, :]

    if (not _HOOK_STATE.enabled) or (not isinstance(module, (Qwen3Attention, Qwen3_5Attention))):
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )

    layer_idx = int(module.layer_idx)
    cached_pages = _HOOK_STATE.pages_by_layer.get(layer_idx, None)
    spec = _HOOK_STATE.spec

    out, st = paged_bitplane_attention_decode(
        query=query,
        key=key,
        value=value,
        scaling=scaling,
        attention_mask=attention_mask,
        mode=spec.mode,
        page_size=PAGE_SIZE,
        recent_tokens=RECENT_TOKENS,
        hist_len=_HOOK_STATE.hist_len,
        top_ratio=spec.top_ratio,
        routing=spec.routing,
        gqa_shared=spec.gqa_shared,
        pages_cache=cached_pages,
    )
    _HOOK_STATE.step_stats.append(
        {
            "layer_idx": float(layer_idx),
            "union_ratio": float(st["union_ratio"]),
            "union_inflation": float(st["union_inflation"]),
            "eff_bits_per_elem": float(st["eff_bits_per_elem"]),
        }
    )
    return out, None


ALL_ATTENTION_FUNCTIONS.register("bitplane_probe", _bitplane_attention_interface)


def _bind_probe(model: torch.nn.Module, impl_name: str) -> None:
    """绑定模型及所有文本注意力子模块的注意力后端名称。"""
    if hasattr(model, "config") and model.config is not None:
        model.config._attn_implementation = impl_name
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            model.config.text_config._attn_implementation = impl_name
    for mod in model.modules():
        if hasattr(mod, "config") and mod.config is not None:
            mod.config._attn_implementation = impl_name


def _load_model(model_name: str, is_qwen3_5: bool, device: torch.device) -> tuple[Any, Any, list[int]]:
    """加载指定模型（bfloat16 上 GPU），并返回全注意力层索引列表。"""
    m_src = _resolve_model_source(model_name)
    tokenizer = AutoTokenizer.from_pretrained(m_src)
    if is_qwen3_5:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(m_src, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(m_src, dtype=torch.bfloat16)
    model.eval()
    model.to(device)

    attn_layers: list[int] = []
    for mod in model.modules():
        if isinstance(mod, (Qwen3Attention, Qwen3_5Attention)):
            attn_layers.append(int(mod.layer_idx))
    attn_layers = sorted(set(attn_layers))
    return model, tokenizer, attn_layers


def _extract_full_attn_kv_from_pkv(
    pkv: Any,
    attn_layers: list[int],
    hist_len: int,
) -> tuple[dict[int, QuantizedBitPlanePages], dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """
    从已预填充的 DynamicCache 提取各全局注意力层前缀 KV，预计算 QuantizedBitPlanePages。
    兼容 transformers 5.x (.layers[i].keys/.values) 与传统 (.key_cache/.value_cache) 接口。
    """
    pages_by_layer: dict[int, QuantizedBitPlanePages] = {}
    raw_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for l_idx in attn_layers:
        if hasattr(pkv, "layers"):
            layer_obj = pkv.layers[l_idx]
            k_t = layer_obj.keys
            v_t = layer_obj.values
        else:
            k_t = pkv.key_cache[l_idx]
            v_t = pkv.value_cache[l_idx]
        # 确保只取前 hist_len 部分建纯前缀静态页表
        pages_by_layer[l_idx] = _build_quantized_pages(k_t, v_t, hist_len=hist_len, page_size=PAGE_SIZE)
        raw_kv[l_idx] = (k_t.detach(), v_t.detach())
    return pages_by_layer, raw_kv


def _build_ppl_sequences(tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    """读取长文档语料构造 2 条不自重复的 L=2048 前缀 + 128 步真实续写样本。"""
    doc1_text, doc2_text = _load_doc_pair()

    total_need = PREFIX_LEN + CONT_STEPS
    seqs: list[dict[str, Any]] = []
    for name, txt in [("PROOF.md", doc1_text), ("RESEARCH_REPORTS.md", doc2_text)]:
        ids = tokenizer.encode(txt, add_special_tokens=False)
        while len(ids) < total_need + 64:
            ids = ids + ids
        # 跳过开头目录行约 64 token
        chunk = ids[64 : 64 + total_need]
        prefix_ids = torch.tensor([chunk[:PREFIX_LEN]], dtype=torch.long, device=device)
        cont_ids = torch.tensor([chunk[PREFIX_LEN:]], dtype=torch.long, device=device)
        seqs.append({"name": name, "prefix_ids": prefix_ids, "cont_ids": cont_ids})
    return seqs


def _build_needle_sequences(tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    """构造 5 个深度 (10%, 30%, 50%, 70%, 90%)、严格 L=2048 的大海捞针检索样本。"""
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
    cases: list[dict[str, Any]] = []
    for name, depth, code, proj in specs:
        needle_str = f" [AUDIT RECORD] The official verification code for {proj} is {code}. "
        # 严格不在末尾加尾随空格，保证 BPE 词边界为 " ORN..." 与完整自回归匹配
        question_str = f" Based strictly on the audit record above, the official verification code for {proj} is"
        n_ids = tokenizer.encode(needle_str, add_special_tokens=False)
        q_ids = tokenizer.encode(question_str, add_special_tokens=False)
        full_q_code = tokenizer.encode(question_str + " " + code, add_special_tokens=False)
        code_ids = full_q_code[len(q_ids):] if full_q_code[:len(q_ids)] == q_ids else tokenizer.encode(" " + code, add_special_tokens=False)
        rem = PREFIX_LEN - len(n_ids) - len(q_ids)
        b_len = max(16, int(round(rem * depth)))
        a_len = rem - b_len

        before_ids: list[int] = []
        while len(before_ids) < b_len:
            before_ids.extend(f_ids)
        before_ids = before_ids[:b_len]

        after_ids: list[int] = []
        while len(after_ids) < a_len:
            after_ids.extend(f_ids)
        after_ids = after_ids[:a_len]

        full_ids = before_ids + n_ids + after_ids + q_ids
        assert len(full_ids) == PREFIX_LEN
        cases.append(
            {
                "name": name,
                "depth": depth,
                "code": code,
                "code_ids": code_ids,
                "input_ids": torch.tensor([full_ids], dtype=torch.long, device=device),
            }
        )
    return cases


def _eval_ppl_and_kl_for_spec(
    model: Any,
    pkv_prefix_minus_1: Any,
    eval_chunk: torch.Tensor,
    target_ids: np.ndarray,
    pages_by_layer: dict[int, QuantizedBitPlanePages],
    spec: MethodSpec,
    ref_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """在固定前缀缓存 pkv_prefix_minus_1 上评测 128 步连续未知续写的真实 PPL、KL 与带宽。"""
    _HOOK_STATE.enabled = True
    _HOOK_STATE.spec = spec
    _HOOK_STATE.hist_len = HIST_LEN
    _HOOK_STATE.pages_by_layer = pages_by_layer
    _HOOK_STATE.reset_stats()

    pkv_copy = copy.deepcopy(pkv_prefix_minus_1)
    with torch.no_grad():
        out = model(input_ids=eval_chunk, past_key_values=pkv_copy, use_cache=True)
    _HOOK_STATE.enabled = False

    logits_np = out.logits[0].detach().float().cpu().numpy()  # (128, vocab)
    max_z = np.max(logits_np, axis=1, keepdims=True)
    lse = max_z[:, 0] + np.log(np.sum(np.exp(logits_np - max_z), axis=1))
    nll = lse - logits_np[np.arange(target_ids.shape[0]), target_ids]
    ppl = float(np.exp(np.mean(nll)))

    kl_mean = 0.0
    if ref_logits is not None:
        max_r = np.max(ref_logits, axis=1, keepdims=True)
        log_p_r = ref_logits - max_r - np.log(np.sum(np.exp(ref_logits - max_r), axis=1, keepdims=True))
        log_p_c = logits_np - max_z - np.log(np.sum(np.exp(logits_np - max_z), axis=1, keepdims=True))
        p_r = np.exp(log_p_r)
        kl_mean = float(np.mean(np.maximum(0.0, np.sum(p_r * (log_p_r - log_p_c), axis=1))))

    stats = _HOOK_STATE.step_stats
    mean_bits = float(np.mean([s["eff_bits_per_elem"] for s in stats])) if stats else 16.0
    mean_union = float(np.mean([s["union_ratio"] for s in stats])) if stats else 1.0
    mean_infl = float(np.mean([s["union_inflation"] for s in stats])) if stats else 1.0

    return {
        "ppl": ppl,
        "kl": kl_mean,
        "eff_bits_per_elem": mean_bits,
        "union_ratio": mean_union,
        "union_inflation": mean_infl,
        "logits": logits_np,
    }


def _eval_needle_for_spec(
    model: Any,
    tokenizer: Any,
    pkv_prefix_minus_1: Any,
    last_prefix_tok: torch.Tensor,
    target_code: str,
    code_ids: list[int],
    pages_by_layer: dict[int, QuantizedBitPlanePages],
    spec: MethodSpec,
) -> dict[str, Any]:
    """在固定前缀缓存上执行 16 步自回归贪心解码（同时做 teacher-forced 密钥均值概率校验）。"""
    _HOOK_STATE.enabled = True
    _HOOK_STATE.spec = spec
    _HOOK_STATE.hist_len = HIST_LEN
    _HOOK_STATE.pages_by_layer = pages_by_layer
    _HOOK_STATE.reset_stats()

    # 1) 贪心自回归生成 16 步检验精确字符串命中
    pkv_gen = copy.deepcopy(pkv_prefix_minus_1)
    cur_tok = last_prefix_tok
    gen_ids: list[int] = []
    with torch.no_grad():
        for _ in range(NEEDLE_DECODE_STEPS):
            out = model(input_ids=cur_tok, past_key_values=pkv_gen, use_cache=True)
            pkv_gen = out.past_key_values
            nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            gen_ids.append(int(nxt.item()))
            cur_tok = nxt
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    hit = target_code in decoded

    # 2) Teacher-forced 计算目标密钥各 token 的真实预测概率均值
    pkv_tf = copy.deepcopy(pkv_prefix_minus_1)
    tf_chunk = torch.cat(
        [last_prefix_tok, torch.tensor([code_ids[:-1]], dtype=torch.long, device=last_prefix_tok.device)],
        dim=1,
    )
    with torch.no_grad():
        tf_out = model(input_ids=tf_chunk, past_key_values=pkv_tf, use_cache=True)
    _HOOK_STATE.enabled = False

    tf_probs = torch.softmax(tf_out.logits[0].float(), dim=-1)  # (len(code_ids), vocab)
    tok_probs = [float(tf_probs[i, cid].item()) for i, cid in enumerate(code_ids)]
    mean_code_prob = float(np.mean(tok_probs))

    return {
        "hit": bool(hit),
        "mean_code_prob": mean_code_prob,
        "decoded": decoded,
    }


def _evaluate_model_suite(
    model_name: str,
    is_qwen3_5: bool,
    device: torch.device,
) -> dict[str, Any]:
    """在单款模型上完整执行基线校准与任务 1 / 2 / 3。"""
    print(f"\n==================== 加载模型: {model_name} ====================", flush=True)
    t_start = time.perf_counter()
    model, tokenizer, attn_layers = _load_model(model_name, is_qwen3_5, device)
    print(f"已加载至 {device} | 全注意力层 ({len(attn_layers)} 层): {attn_layers}", flush=True)

    ppl_seqs = _build_ppl_sequences(tokenizer, device)
    needle_seqs = _build_needle_sequences(tokenizer, device)

    # 先在原生注意力实现下做前向基线对齐校验（PPL & logits 差）
    _bind_probe(model, "eager")
    _HOOK_STATE.enabled = False

    ppl_prepared: list[dict[str, Any]] = []
    max_sanity_diff = 0.0
    max_sanity_ppl_diff = 0.0
    layer_outlier_accum: dict[int, list[dict[str, float]]] = {l: [] for l in attn_layers}

    for item in ppl_seqs:
        prefix_ids = item["prefix_ids"]
        cont_ids = item["cont_ids"]
        eval_chunk = torch.cat([prefix_ids[:, -1:], cont_ids[:, :-1]], dim=1)  # (1, 128)
        targets = cont_ids[0].detach().cpu().numpy()

        # 原生 eager 参考运行
        with torch.no_grad():
            pkv_nat = model(input_ids=prefix_ids[:, :-1], use_cache=True).past_key_values
            out_nat = model(input_ids=eval_chunk, past_key_values=copy.deepcopy(pkv_nat), use_cache=True)
        nat_logits = out_nat.logits[0].detach().float().cpu().numpy()
        nat_max_z = np.max(nat_logits, axis=1, keepdims=True)
        nat_lse = nat_max_z[:, 0] + np.log(np.sum(np.exp(nat_logits - nat_max_z), axis=1))
        nat_ppl = float(np.exp(np.mean(nat_lse - nat_logits[np.arange(targets.shape[0]), targets])))

        # 绑定探针接口并预建 QuantizedBitPlanePages + 层级统计
        _bind_probe(model, "bitplane_probe")
        _HOOK_STATE.enabled = False
        with torch.no_grad():
            pkv_probe = model(input_ids=prefix_ids[:, :-1], use_cache=True).past_key_values
        pages_by_layer, raw_kv = _extract_full_attn_kv_from_pkv(pkv_probe, attn_layers, hist_len=HIST_LEN)

        for l_idx in attn_layers:
            k_l, v_l = raw_kv[l_idx]
            layer_outlier_accum[l_idx].append(compute_layer_outlier_stats(k_l, v_l, page_size=PAGE_SIZE))

        fp16_res = _eval_ppl_and_kl_for_spec(
            model, pkv_probe, eval_chunk, targets, pages_by_layer, MethodSpec("fp16", "fp16"), ref_logits=None
        )
        logit_diff = float(np.max(np.abs(nat_logits - fp16_res["logits"])))
        ppl_diff = abs(nat_ppl - fp16_res["ppl"])
        max_sanity_diff = max(max_sanity_diff, logit_diff)
        max_sanity_ppl_diff = max(max_sanity_ppl_diff, ppl_diff)
        print(
            f"  [基线自检 {item['name']}] Native PPL={nat_ppl:.4f} vs Probe-FP16 PPL={fp16_res['ppl']:.4f} "
            f"| max_logit_diff={logit_diff:.4e}",
            flush=True,
        )

        # 还原回 eager 备下个样本原生校验
        _bind_probe(model, "eager")

        ppl_prepared.append(
            {
                "name": item["name"],
                "pkv": pkv_probe,
                "eval_chunk": eval_chunk,
                "targets": targets,
                "pages": pages_by_layer,
                "ref_logits": fp16_res["logits"],
                "ref_ppl": fp16_res["ppl"],
            }
        )

    # 绑定为 bitplane_probe 开始完整评测
    _bind_probe(model, "bitplane_probe")

    # 1) 任务 1：PPL & KL & 实际带宽评测
    print("  [任务 1] 运行 14 种方法 L=2048 + 128 步续写 PPL 评测...", flush=True)
    ppl_summary: dict[str, Any] = {}
    for spec in PPL_METHODS:
        ppls, kls, bits_l, unions_l, infls_l = [], [], [], [], []
        for prep in ppl_prepared:
            r = _eval_ppl_and_kl_for_spec(
                model,
                prep["pkv"],
                prep["eval_chunk"],
                prep["targets"],
                prep["pages"],
                spec,
                ref_logits=prep["ref_logits"],
            )
            ppls.append(r["ppl"])
            kls.append(r["kl"])
            bits_l.append(r["eff_bits_per_elem"])
            unions_l.append(r["union_ratio"])
            infls_l.append(r["union_inflation"])

        fp16_mean_ppl = float(np.mean([p["ref_ppl"] for p in ppl_prepared]))
        m_ppl = float(np.mean(ppls))
        ppl_summary[spec.name] = {
            "mean_ppl": m_ppl,
            "delta_ppl": m_ppl - fp16_mean_ppl,
            "mean_kl": float(np.mean(kls)),
            "eff_bits_per_elem": float(np.mean(bits_l)),
            "union_ratio": float(np.mean(unions_l)),
            "union_inflation": float(np.mean(infls_l)),
        }
        print(
            f"    {spec.name:22s} | bits={ppl_summary[spec.name]['eff_bits_per_elem']:5.2f} "
            f"(infl={ppl_summary[spec.name]['union_inflation']:.2f}x) | "
            f"PPL={m_ppl:7.3f} (dPPL={m_ppl - fp16_mean_ppl:+6.3f}) | "
            f"KL={ppl_summary[spec.name]['mean_kl']:.5f}",
            flush=True,
        )

    # 2) 任务 2：5 深度大海捞针精准检索
    print("  [任务 2] 准备并运行 5 深度 (10%–90%, L=2048) 大海捞针评测...", flush=True)
    _HOOK_STATE.enabled = False
    needle_prepared: list[dict[str, Any]] = []
    for nc in needle_seqs:
        with torch.no_grad():
            pkv_n = model(input_ids=nc["input_ids"][:, :-1], use_cache=True).past_key_values
        pages_n, _ = _extract_full_attn_kv_from_pkv(pkv_n, attn_layers, hist_len=HIST_LEN)
        needle_prepared.append(
            {
                "name": nc["name"],
                "depth": nc["depth"],
                "code": nc["code"],
                "code_ids": nc["code_ids"],
                "pkv": pkv_n,
                "last_tok": nc["input_ids"][:, -1:],
                "pages": pages_n,
            }
        )

    needle_summary: dict[str, Any] = {}
    for spec in NEEDLE_METHODS:
        hits = 0
        probs_all = []
        per_case = []
        for np_item in needle_prepared:
            nr = _eval_needle_for_spec(
                model,
                tokenizer,
                np_item["pkv"],
                np_item["last_tok"],
                np_item["code"],
                np_item["code_ids"],
                np_item["pages"],
                spec,
            )
            hits += int(nr["hit"])
            probs_all.append(nr["mean_code_prob"])
            per_case.append(
                {
                    "depth": np_item["depth"],
                    "hit": nr["hit"],
                    "mean_code_prob": nr["mean_code_prob"],
                    "decoded": nr["decoded"],
                }
            )
        needle_summary[spec.name] = {
            "hits": hits,
            "total": len(needle_prepared),
            "hit_rate": hits / len(needle_prepared),
            "mean_code_prob": float(np.mean(probs_all)),
            "cases": per_case,
        }
        print(
            f"    {spec.name:22s} | hits={hits}/{len(needle_prepared)} | "
            f"mean_code_prob={needle_summary[spec.name]['mean_code_prob']:.4f}",
            flush=True,
        )

    # 3) 任务 3：逐层 K/V 离群与量化敏感度汇总
    layer_diag: dict[str, Any] = {}
    for l_idx in attn_layers:
        rows = layer_outlier_accum[l_idx]
        layer_diag[str(l_idx)] = {
            k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()
        }

    # 释放 GPU 显存以免切换模型冲突
    _HOOK_STATE.pages_by_layer.clear()
    _HOOK_STATE.reset_stats()
    del model, ppl_prepared, needle_prepared
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_start
    print(f"模型 {model_name} 全部评测完成，耗时 {elapsed:.1f}s。", flush=True)
    return {
        "model_name": model_name,
        "architecture": "Hybrid (18 GatedDeltaNet + 6 4:1 GQA Global)" if is_qwen3_5 else "Dense (28 2:1 GQA Global)",
        "full_attn_layers": attn_layers,
        "sanity_max_logit_diff": max_sanity_diff,
        "sanity_max_ppl_diff": max_sanity_ppl_diff,
        "ppl_results": ppl_summary,
        "needle_results": needle_summary,
        "layer_diagnostics": layer_diag,
        "elapsed_sec": elapsed,
    }


def _plot_kill_test_figure(res: dict[str, Any], out_path: Path) -> None:
    """绘制 2x2 300 DPI 无遮挡英文学术图表总结 Kill Test 全部核心发现。"""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.4))
    q35 = res["models"]["Qwen3.5-0.8B"]
    q3 = res["models"]["Qwen3-0.6B"]

    # --- Panel (a): Qwen3.5-0.8B (Hybrid) Bandwidth vs Delta PPL Pareto 前沿 ---
    ax_a = axes[0, 0]
    p35 = q35["ppl_results"]

    # Bit-Plane 曲线 (10%, 15%, 25%) 与 Base2 (0%)、INT4 (100%) 连接展现 Pareto 轨迹
    bp_keys = ["base2", "bitplane_10_max", "bitplane_15_max", "bitplane_25_max", "int4"]
    bp_bits = [p35[k]["eff_bits_per_elem"] for k in bp_keys]
    bp_dppl = [p35[k]["delta_ppl"] for k in bp_keys]
    ax_a.plot(
        bp_bits,
        bp_dppl,
        "-o",
        color="#0072B2",
        linewidth=2.2,
        markersize=7.0,
        label="BitPlane (2-bit Base + Top-k Resid, GQA)",
        zorder=4,
    )

    single_pts = [
        ("int2", "Uniform INT2 (2.0b)", "#D55E00", "X", 8.5),
        ("bitplane_15_mass", "BitPlane-15% (Mass)", "#009E73", "D", 7.2),
        ("bitplane_15_droptail", "Ablation: DropTail-15%", "#CC79A7", "v", 7.5),
        ("quest_10_gqa", "Quest-10% (GQA)", "#E69F00", "s", 7.0),
        ("quest_15_gqa", "Quest-15% (GQA)", "#E69F00", "^", 7.2),
        ("quest_15_perhead", "Quest-15% (PerHead Actual)", "#7F7F7F", "P", 7.8),
    ]
    for k, lbl, col, mk, ms in single_pts:
        ax_a.scatter(
            p35[k]["eff_bits_per_elem"],
            p35[k]["delta_ppl"],
            color=col,
            marker=mk,
            s=ms * 10.0,
            label=lbl,
            edgecolors="#222222",
            linewidths=0.6,
            zorder=5,
        )
    ax_a.axhline(0.0, color="#111111", linestyle="--", linewidth=1.2, label="FP16 Baseline (dPPL=0)")
    ax_a.set_xlabel("Effective HBM KV Read Bandwidth (bits / element)")
    ax_a.set_ylabel("Continuation Delta PPL vs FP16 (Lower = Better)")
    ax_a.set_title("(a) Qwen3.5-0.8B (Hybrid): Bandwidth vs Delta PPL Pareto")
    ax_a.legend(loc="upper right", ncol=2, fontsize=7.2, frameon=True, framealpha=0.94)

    # --- Panel (b): 五深度 (10%–90%) 大海捞针命中率对比 (Hybrid vs Dense) ---
    ax_b = axes[0, 1]
    cmp_methods = [
        ("fp16", "FP16"),
        ("int4", "INT4"),
        ("int2", "INT2"),
        ("base2", "Base2"),
        ("quest_15_gqa", "Quest-15%\n(GQA)"),
        ("bitplane_15_droptail", "DropTail\n15%"),
        ("bitplane_15_max", "BitPlane\n15% Max"),
        ("bitplane_15_mass", "BitPlane\n15% Mass"),
    ]
    x_idx = np.arange(len(cmp_methods))
    w = 0.36
    q35_hits = [q35["needle_results"][k]["hit_rate"] * 100.0 for k, _ in cmp_methods]
    q3_hits = [q3["needle_results"][k]["hit_rate"] * 100.0 for k, _ in cmp_methods]

    b1 = ax_b.bar(x_idx - w / 2, q35_hits, width=w, color="#0072B2", edgecolor="#222222", alpha=0.9, label="Qwen3.5-0.8B (Hybrid 3:1)")
    b2 = ax_b.bar(x_idx + w / 2, q3_hits, width=w, color="#009E73", edgecolor="#222222", alpha=0.9, label="Qwen3-0.6B (Dense)")
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax_b.text(rect.get_x() + rect.get_width() / 2.0, h + 1.5, f"{int(round(h))}%", ha="center", va="bottom", fontsize=7.0)
    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels([lbl for _, lbl in cmp_methods], fontsize=7.6)
    ax_b.set_ylim(0.0, 118.0)
    ax_b.set_ylabel("5-Depth Needle Recall Accuracy (%)")
    ax_b.set_title("(b) L=2048 5-Depth Needle-in-a-Haystack Retrieval")
    ax_b.legend(loc="upper left", fontsize=8.0, frameon=True, framealpha=0.94)

    # --- Panel (c): GQA 页并集膨胀效应 (名义带宽 vs 实际 HBM 加载带宽) ---
    ax_c = axes[1, 0]
    bw_items = [
        ("INT4\n(Static)", 4.0, 4.0),
        ("Quest-15%\n(PerHead 2:1)", 2.4, q3["ppl_results"]["quest_15_perhead"]["eff_bits_per_elem"]),
        ("Quest-15%\n(PerHead 4:1)", 2.4, q35["ppl_results"]["quest_15_perhead"]["eff_bits_per_elem"]),
        ("Quest-15%\n(GQA Shared)", 2.4, q35["ppl_results"]["quest_15_gqa"]["eff_bits_per_elem"]),
        ("BitPlane-15%\n(GQA Shared)", 2.3, q35["ppl_results"]["bitplane_15_max"]["eff_bits_per_elem"]),
    ]
    xb = np.arange(len(bw_items))
    nom_vals = [it[1] for it in bw_items]
    act_vals = [it[2] for it in bw_items]
    r_nom = ax_c.bar(xb - w / 2, nom_vals, width=w, color="#A6CEE3", edgecolor="#222222", label="Nominal Per-Head Read (bits)")
    r_act = ax_c.bar(xb + w / 2, act_vals, width=w, color="#D55E00", edgecolor="#222222", label="Actual GQA HBM Union Read (bits)")
    for rect in list(r_nom) + list(r_act):
        h = rect.get_height()
        ax_c.text(rect.get_x() + rect.get_width() / 2.0, h + 0.12, f"{h:.2f}", ha="center", va="bottom", fontsize=7.4)
    ax_c.set_xticks(xb)
    ax_c.set_xticklabels([it[0] for it in bw_items], fontsize=7.8)
    ax_c.set_ylim(0.0, max(act_vals) * 1.25)
    ax_c.set_ylabel("HBM Read Bandwidth (bits / element)")
    ax_c.set_title("(c) GQA Union Page Inflation vs GQA-Shared BitPlane")
    ax_c.legend(loc="upper right", fontsize=8.0, frameon=True, framealpha=0.94)

    # --- Panel (d): Hybrid 稀缺全局层 vs Dense 稠密层 K/V 离群峰值与 INT2 KL 敏感度 (双纵轴) ---
    ax_d = axes[1, 1]
    diag_names = ["Key-Only INT2\nKL (Hybrid vs Dense)", "Val-Only INT2\nKL (Hybrid vs Dense)"]
    xd = np.arange(len(diag_names))
    w_d = 0.34
    kl35 = [
        q35["ppl_results"]["key_only_int2"]["mean_kl"],
        q35["ppl_results"]["val_only_int2"]["mean_kl"],
    ]
    kl3 = [
        q3["ppl_results"]["key_only_int2"]["mean_kl"],
        q3["ppl_results"]["val_only_int2"]["mean_kl"],
    ]
    bd1 = ax_d.bar(xd - w_d / 2, kl35, width=w_d, color="#0072B2", edgecolor="#222222", alpha=0.9, label="Qwen3.5-0.8B KL (Hybrid 6 Global)")
    bd2 = ax_d.bar(xd + w_d / 2, kl3, width=w_d, color="#009E73", edgecolor="#222222", alpha=0.9, label="Qwen3-0.6B KL (Dense 28 Global)")
    for rect in list(bd1) + list(bd2):
        h = rect.get_height()
        ax_d.text(rect.get_x() + rect.get_width() / 2.0, h + 0.003, f"{h:.4f}", ha="center", va="bottom", fontsize=7.6)

    ax_d.set_xticks([0, 1, 2, 3])
    ax_d.set_xticklabels(["Key INT2\nKL Div", "Val INT2\nKL Div", "Mean Key\nPeak/RMS", "Mean Val\nPeak/RMS"], fontsize=8.0)
    ax_d.set_ylim(0.0, max(kl35 + kl3) * 1.32)
    ax_d.set_ylabel("Continuation KL Divergence (Left Axis)", color="#111111")

    ax_d2 = ax_d.twinx()
    ax_d2.spines["top"].set_visible(False)
    ax_d2.grid(False)
    xd_r = np.array([2, 3])
    pk35 = [
        float(np.mean([v["k_peak_over_rms"] for v in q35["layer_diagnostics"].values()])),
        float(np.mean([v["v_peak_over_rms"] for v in q35["layer_diagnostics"].values()])),
    ]
    pk3 = [
        float(np.mean([v["k_peak_over_rms"] for v in q3["layer_diagnostics"].values()])),
        float(np.mean([v["v_peak_over_rms"] for v in q3["layer_diagnostics"].values()])),
    ]
    br1 = ax_d2.bar(xd_r - w_d / 2, pk35, width=w_d, color="#56B4E9", edgecolor="#222222", hatch="//", alpha=0.9, label="Qwen3.5 Peak/RMS (Right)")
    br2 = ax_d2.bar(xd_r + w_d / 2, pk3, width=w_d, color="#A6D854", edgecolor="#222222", hatch="//", alpha=0.9, label="Qwen3 Peak/RMS (Right)")
    for rect in list(br1) + list(br2):
        h = rect.get_height()
        ax_d2.text(rect.get_x() + rect.get_width() / 2.0, h + 0.6, f"{h:.1f}x", ha="center", va="bottom", fontsize=7.6)
    ax_d2.set_ylim(0.0, max(pk35 + pk3) * 1.32)
    ax_d2.set_ylabel("Max |x| / RMS Outlier Ratio (Right Axis)", color="#333333")
    ax_d.axvline(1.5, color="#888888", linestyle=":", linewidth=1.1)
    ax_d.set_title("(d) Hybrid vs Dense: K/V INT2 KL & Outlier Ratio")

    h1, l1 = ax_d.get_legend_handles_labels()
    h2, l2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.1, frameon=True, framealpha=0.94)

    fig.tight_layout(h_pad=2.4, w_pad=2.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU！")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"启动 Bit-Plane Paged Attention Kill Test | GPU={props.name} | Python={platform.python_version()}")

    res_q35 = _evaluate_model_suite("Qwen3.5-0.8B", is_qwen3_5=True, device=device)
    res_q3 = _evaluate_model_suite("Qwen3-0.6B", is_qwen3_5=False, device=device)

    all_res = {
        "hardware": {
            "gpu": props.name,
            "total_vram_gb": props.total_memory / (1024 ** 3),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "config": {
            "prefix_len": PREFIX_LEN,
            "page_size": PAGE_SIZE,
            "recent_tokens": RECENT_TOKENS,
            "hist_len": HIST_LEN,
            "cont_steps": CONT_STEPS,
        },
        "models": {
            "Qwen3.5-0.8B": res_q35,
            "Qwen3-0.6B": res_q3,
        },
    }

    OUT_JSON.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_kill_test_figure(all_res, OUT_FIG)
    print(f"\n[DONE] 原始实测结果已写至: {OUT_JSON}")
    print(f"[DONE] 300 DPI 总结图已写至: {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
