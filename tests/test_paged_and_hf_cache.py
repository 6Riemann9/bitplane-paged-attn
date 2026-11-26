"""
真 PagedAttention 物理页池、Prefix Caching (CoW)、Continuous Batching 与端到端 BitPlaneDynamicCache 单测。
"""

from __future__ import annotations

import copy
import gc
import math
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM

from bitplane_attn import (
    BitPlaneCacheLayer,
    BitPlaneDynamicCache,
    PackedBitPlaneKVCache,
    PagedBitPlaneKVPool,
    compute_logical_read_bits_per_kv_elem,
    paged_bitplane_attention_decode,
    triton_bitplane_dualstream_decode,
    triton_int4_paged_decode,
    triton_paged_bitplane_dualstream_decode,
    triton_paged_int4_decode,
)
from bitplane_attn.quant_bitplane import QuantizedBitPlanePages


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU 运行 Triton 核")


def _pool_seq_to_ref_pages(pool: PagedBitPlaneKVPool, seq_id: int) -> QuantizedBitPlanePages:
    """将真页池中的单条序列按物理块映射解包至参考浮点页对象，验证算子算术零误差。"""
    u = pool.unpack_sequence(seq_id)
    seq = pool.sequences[seq_id]
    n_pages = seq.num_pages
    if n_pages == 0:
        empty_m = torch.empty((1, pool.num_kv_heads, 0, pool.head_dim), dtype=torch.float32, device=pool.device)
        return QuantizedBitPlanePages(
            page_size=pool.page_size,
            hist_len=0,
            num_pages=0,
            k_base=u["k_base"],
            k_int4=u["k_int4"],
            k_int2=u["k_base"],
            v_base=u["v_base"],
            v_int4=u["v_int4"],
            v_int2=u["v_base"],
            k_page_min=empty_m,
            k_page_max=empty_m,
        )
    idx_t = torch.tensor(seq.block_ids, dtype=torch.long, device=pool.device)
    km = pool.k_min_pool.index_select(0, idx_t).float().permute(1, 0, 2).unsqueeze(0)
    ks = pool.k_scale_pool.index_select(0, idx_t).float().permute(1, 0, 2).unsqueeze(0)
    return QuantizedBitPlanePages(
        page_size=pool.page_size,
        hist_len=n_pages * pool.page_size,
        num_pages=n_pages,
        k_base=u["k_base"],
        k_int4=u["k_int4"],
        k_int2=u["k_base"],
        v_base=u["v_base"],
        v_int4=u["v_int4"],
        v_int2=u["v_base"],
        k_page_min=km,
        k_page_max=km + 15.0 * ks,
    )


def test_incremental_paging_matches_bulk_from_kv() -> None:
    """验证分批增量 append_kv（跨多次满页封存）与一次性 PackedBitPlaneKVCache.from_kv 逐位一致。"""
    torch.manual_seed(7001)
    h_kv, d, total_len = 4, 128, 173
    k = torch.randn((1, h_kv, total_len, d), device="cuda:0", dtype=torch.float16)
    v = torch.randn((1, h_kv, total_len, d), device="cuda:0", dtype=torch.float16)

    bulk = PackedBitPlaneKVCache.from_kv(k, v, page_size=32, recent_tokens=32)

    pool = PagedBitPlaneKVPool(num_kv_heads=h_kv, head_dim=d, page_size=32, capacity=4, kv_dtype=torch.float16, device="cuda:0")
    sid = pool.allocate_sequence()
    # 先 Prefill 100 token，再逐 token 追加 73 步，期间将两次跨过封页边界并触发物理池自动倍增扩容
    pool.append_kv(sid, k[:, :, :100, :], v[:, :, :100, :], recent_tokens=32)
    for t in range(100, total_len):
        pool.append_kv(sid, k[:, :, t : t + 1, :], v[:, :, t : t + 1, :], recent_tokens=32)

    seq = pool.sequences[sid]
    assert seq.num_pages == bulk.num_pages
    assert seq.tail_len == bulk.tail_len
    assert pool.capacity >= seq.num_pages

    u_pool = pool.unpack_sequence(sid)
    u_bulk = bulk.unpack_to_dense_dequantized()
    assert torch.equal(u_pool["k_int4"], u_bulk["k_int4"])
    assert torch.equal(u_pool["k_base"], u_bulk["k_base"])
    assert torch.equal(u_pool["v_int4"], u_bulk["v_int4"])
    assert torch.equal(seq.k_tail, bulk.k_tail)
    assert torch.equal(seq.v_tail, bulk.v_tail)


def test_prefix_caching_cow_and_zero_block_leak() -> None:
    """验证 Prefix Caching 零拷贝分叉、Copy-on-Write 隔离性与释放后的物理块零泄漏。"""
    torch.manual_seed(7002)
    h_kv, d = 4, 128
    k_pref = torch.randn((1, h_kv, 160, d), device="cuda:0", dtype=torch.float16)
    v_pref = torch.randn((1, h_kv, 160, d), device="cuda:0", dtype=torch.float16)

    pool = PagedBitPlaneKVPool(num_kv_heads=h_kv, head_dim=d, page_size=32, capacity=16, kv_dtype=torch.float16, device="cuda:0")
    init_free = len(pool.free_blocks)
    s0 = pool.allocate_sequence()
    p_keys = [f"prefix-page-{i}".encode("utf-8") for i in range(4)]
    pool.append_kv(s0, k_pref, v_pref, recent_tokens=32, prefix_keys=p_keys)
    assert pool.sequences[s0].num_pages == 4

    # 1) fork_sequence 共享全部 4 个物理页且 ref_count == 2
    s1 = pool.fork_sequence(s0)
    assert pool.sequences[s1].block_ids == pool.sequences[s0].block_ids
    for blk in pool.sequences[s0].block_ids:
        assert pool.ref_counts[blk] == 2
    st_shared = pool.memory_stats()
    assert st_shared["unique_active_blocks"] == 4
    assert st_shared["logical_total_blocks"] == 8
    assert abs(st_shared["prefix_sharing_ratio"] - 2.0) < 1e-6

    # 2) 内容哈希自动 Prefix Caching 去重（再建同 prefix_keys 的新序列 s2）
    s2 = pool.allocate_sequence()
    pool.append_kv(s2, k_pref, v_pref, recent_tokens=32, prefix_keys=p_keys)
    assert pool.sequences[s2].block_ids == pool.sequences[s0].block_ids
    for blk in pool.sequences[s0].block_ids:
        assert pool.ref_counts[blk] == 3

    # 3) 对 s1 的逻辑第 2 页触发 Copy-on-Write 改写，验证源序列 s0/s2 完全隔离不变
    s0_before = pool.unpack_sequence(s0)["k_int4"].clone()
    shared_blk_2 = pool.sequences[s0].block_ids[2]
    k_new_p = torch.randn((1, h_kv, 32, d), device="cuda:0", dtype=torch.float16) * 5.0
    v_new_p = torch.randn((1, h_kv, 32, d), device="cuda:0", dtype=torch.float16) * 5.0
    cow_blk = pool.write_page(s1, 2, k_new_p, v_new_p)
    assert cow_blk != shared_blk_2
    assert pool.ref_counts[shared_blk_2] == 2
    assert pool.ref_counts[cow_blk] == 1
    assert torch.equal(pool.unpack_sequence(s0)["k_int4"], s0_before)
    assert torch.equal(pool.unpack_sequence(s2)["k_int4"], s0_before)

    # 4) 依次释放三条序列，校验全部物理块归还且 prefix_table 清零
    pool.free_sequence(s0)
    assert pool.ref_counts[shared_blk_2] == 1
    pool.free_sequence(s1)
    pool.free_sequence(s2)
    assert len(pool.free_blocks) == pool.capacity == init_free
    assert all(c == 0 for c in pool.ref_counts)
    assert len(pool.prefix_table) == 0


@pytest.mark.parametrize("head_dim,gqa_grp", [(64, 2), (128, 4), (256, 4)])
def test_paged_shuffled_blocks_match_reference(head_dim: int, gqa_grp: int) -> None:
    """制造碎片化乱序物理页表，验证真分页 INT4 与 DualStream 核精确对齐参考实现。"""
    torch.manual_seed(7100 + head_dim + gqa_grp)
    bsz, h_kv = 2, 4
    h_q = h_kv * gqa_grp
    hist_len, tail_len = 512, 19
    lk = hist_len + tail_len
    scaling = 1.0 / math.sqrt(head_dim)

    q = torch.randn((bsz, h_q, 1, head_dim), device="cuda:0", dtype=torch.float16)
    k = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.6
    v = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.6
    k[:, :, 3, :] = q[:, ::gqa_grp, 0, :] * 4.0
    k[:, :, 6 * 32 + 7, :] = q[:, ::gqa_grp, 0, :] * 3.5

    # 为避免均匀页得分并列导致边界选页差 1 页，加轻量单调区分斜坡
    for p_idx in range(hist_len // 32):
        if p_idx not in (0, 6):
            k[:, :, p_idx * 32 : (p_idx + 1) * 32, :] *= 1.0 - 0.025 * p_idx

    pool, sids = PagedBitPlaneKVPool.from_dense_kv(
        k, v, page_size=32, recent_tokens=tail_len, hist_len=hist_len, shuffle_blocks=True, seed=999
    )
    # 确认页表的确是非连续乱序的
    assert pool.sequences[sids[0]].block_ids != list(range(pool.sequences[sids[0]].num_pages))
    batch = pool.build_batch(sids)

    out_i4 = triton_int4_paged_decode(q, batch, scaling)
    out_ds, st_ds = triton_bitplane_dualstream_decode(
        q, batch, scaling, hot_ratio=0.15, warm_ratio=0.35, ellipsoid_beta=1.5, return_stats=True
    )
    assert abs(
        st_ds["logical_read_bits_per_kv_elem"]
        - compute_logical_read_bits_per_kv_elem(
            "bitplane_dualstream",
            hot_ratio=st_ds["hot_ratio"],
            warm_ratio=st_ds["warm_ratio"],
            skip_ratio=st_ds["skip_ratio"],
            head_dim=head_dim,
            page_size=32,
        )
    ) < 1e-6

    # 逐序列对比参考输出
    for b_idx, sid in enumerate(sids):
        ref_pages = _pool_seq_to_ref_pages(pool, sid)
        ref_i4, _ = paged_bitplane_attention_decode(
            q[b_idx : b_idx + 1], k[b_idx : b_idx + 1], v[b_idx : b_idx + 1], scaling,
            mode="int4", page_size=32, recent_tokens=tail_len, hist_len=hist_len, pages_cache=ref_pages,
        )
        ref_ds, _ = paged_bitplane_attention_decode(
            q[b_idx : b_idx + 1], k[b_idx : b_idx + 1], v[b_idx : b_idx + 1], scaling,
            mode="bitplane_dualstream", page_size=32, recent_tokens=tail_len, hist_len=hist_len,
            hot_ratio=0.15, warm_ratio=0.35, ellipsoid_beta=1.5, pages_cache=ref_pages,
        )
        err_i4 = float((out_i4[b_idx : b_idx + 1].float() - ref_i4.float()).abs().max().item())
        diff_ds = (out_ds[b_idx : b_idx + 1].float() - ref_ds.float()).abs()
        rel_ds = float(diff_ds.norm().item() / max(float(ref_ds.float().norm().item()), 1e-12))
        assert err_i4 < 2e-3, f"乱序物理页 INT4 误差超限: {err_i4}"
        assert rel_ds < 1e-3, f"乱序物理页 DualStream rel_l2={rel_ds:.5e}"


@pytest.mark.parametrize("score_mode", ["anisotropic", "center", "box_l1"])
def test_paged_score_modes_match_reference(score_mode: str) -> None:
    """验证三种零额外元数据页排序策略 (anisotropic / center / box_l1) 精准对齐参考实现。"""
    torch.manual_seed(7200)
    bsz, h_kv, grp, head_dim = 2, 4, 4, 128
    h_q = h_kv * grp
    hist_len, tail_len = 512, 15
    lk = hist_len + tail_len
    scaling = 1.0 / math.sqrt(head_dim)

    q = torch.randn((bsz, h_q, 1, head_dim), device="cuda:0", dtype=torch.float16)
    k = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.5
    v = torch.randn((bsz, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.5
    for p_idx in range(hist_len // 32):
        k[:, :, p_idx * 32 : (p_idx + 1) * 32, :] *= 1.0 - 0.03 * p_idx

    pool, sids = PagedBitPlaneKVPool.from_dense_kv(
        k, v, page_size=32, recent_tokens=tail_len, hist_len=hist_len, shuffle_blocks=True
    )
    batch = pool.build_batch(sids)
    beta = 0.0 if score_mode == "center" else 1.5
    out_t = triton_paged_bitplane_dualstream_decode(
        q, batch, scaling, hot_ratio=0.15, warm_ratio=0.35, beta=beta, score_mode=score_mode
    )
    for b_idx, sid in enumerate(sids):
        ref_pages = _pool_seq_to_ref_pages(pool, sid)
        out_r, _ = paged_bitplane_attention_decode(
            q[b_idx : b_idx + 1], k[b_idx : b_idx + 1], v[b_idx : b_idx + 1], scaling,
            mode="bitplane_dualstream", page_size=32, recent_tokens=tail_len, hist_len=hist_len,
            hot_ratio=0.15, warm_ratio=0.35, ellipsoid_beta=beta, score_mode=score_mode, pages_cache=ref_pages,
        )
        diff = (out_t[b_idx : b_idx + 1].float() - out_r.float()).abs()
        rel_l2 = float(diff.norm().item() / max(float(out_r.float().norm().item()), 1e-12))
        assert rel_l2 < 1e-3, f"score_mode={score_mode} rel_l2={rel_l2:.5e}"


def test_continuous_batching_variable_lengths() -> None:
    """验证高度异构长度批处理 [512+17, 128+0, 0+29, 320+32] 对齐单样本真值。"""
    torch.manual_seed(7300)
    h_kv, grp, head_dim = 4, 4, 128
    h_q = h_kv * grp
    scaling = 1.0 / math.sqrt(head_dim)
    specs = [(512, 17), (128, 0), (0, 29), (320, 32)]
    bsz = len(specs)

    q = torch.randn((bsz, h_q, 1, head_dim), device="cuda:0", dtype=torch.float16)
    pool = PagedBitPlaneKVPool(num_kv_heads=h_kv, head_dim=head_dim, page_size=32, capacity=48, kv_dtype=torch.float16, device="cuda:0")
    pool.shuffle_free_blocks(seed=321)

    sids: list[int] = []
    raw_kv: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    for h_len, t_len in specs:
        lk = h_len + t_len
        ks = torch.randn((1, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.5
        vs = torch.randn((1, h_kv, lk, head_dim), device="cuda:0", dtype=torch.float16) * 0.5
        for p_idx in range(h_len // 32):
            ks[:, :, p_idx * 32 : (p_idx + 1) * 32, :] *= 1.0 - 0.04 * p_idx
        sid = pool.allocate_sequence()
        pool.append_kv(sid, ks, vs, recent_tokens=t_len)
        sids.append(sid)
        raw_kv.append((ks, vs, h_len, t_len))

    batch = pool.build_batch(sids)
    out_i4 = triton_paged_int4_decode(q, batch, scaling)
    out_ds = triton_paged_bitplane_dualstream_decode(q, batch, scaling, hot_ratio=0.15, warm_ratio=0.35, beta=1.5)

    for b_idx, (sid, (ks, vs, h_len, t_len)) in enumerate(zip(sids, raw_kv)):
        ref_pages = _pool_seq_to_ref_pages(pool, sid)
        r_i4, _ = paged_bitplane_attention_decode(
            q[b_idx : b_idx + 1], ks, vs, scaling,
            mode="int4", page_size=32, recent_tokens=max(1, t_len), hist_len=h_len, pages_cache=ref_pages,
        )
        r_ds, _ = paged_bitplane_attention_decode(
            q[b_idx : b_idx + 1], ks, vs, scaling,
            mode="bitplane_dualstream", page_size=32, recent_tokens=max(1, t_len), hist_len=h_len,
            hot_ratio=0.15, warm_ratio=0.35, ellipsoid_beta=1.5, pages_cache=ref_pages,
        )
        d_i4 = float((out_i4[b_idx : b_idx + 1].float() - r_i4.float()).abs().max().item())
        diff_ds = (out_ds[b_idx : b_idx + 1].float() - r_ds.float()).abs()
        rel_ds = float(diff_ds.norm().item() / max(float(r_ds.float().norm().item()), 1e-12))
        assert d_i4 < 2e-3, f"样本 {b_idx} (hist={h_len}, tail={t_len}) INT4 误差={d_i4}"
        assert rel_ds < 1e-3, f"样本 {b_idx} (hist={h_len}, tail={t_len}) DualStream rel_l2={rel_ds:.5e}"


def test_hf_bitplane_dynamic_cache_generate_and_vram_savings() -> None:
    """使用无外部权重依赖的合成 Qwen3 模型验证 BitPlaneDynamicCache generate 端到端与真 GPU VRAM 压缩。"""
    torch.manual_seed(7400)
    device = torch.device("cuda:0")
    cfg = Qwen3Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=2048,
    )
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    bsz, l_prefill, gen_steps = 2, 1024, 40
    input_ids = torch.randint(0, 500, (bsz, l_prefill), device=device, dtype=torch.long)

    # 1) 原生 DynamicCache 占用
    gc.collect()
    torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated(device)
    native_cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out_nat = model.generate(
            input_ids=input_ids,
            cache=native_cache,
            use_cache=True,
            max_new_tokens=gen_steps,
            do_sample=False,
        )
    nat_kv_bytes = torch.cuda.memory_allocated(device) - base_mem
    assert out_nat.shape == (bsz, l_prefill + gen_steps)
    del native_cache, out_nat
    gc.collect()
    torch.cuda.empty_cache()

    # 2) BitPlaneDynamicCache 真页池生成与物理显存实测
    base_mem_bp = torch.cuda.memory_allocated(device)
    bp_cache = BitPlaneDynamicCache(
        config=model.config,
        page_size=32,
        recent_tokens=32,
        hot_ratio=0.15,
        warm_ratio=0.35,
        collect_stats=True,
    )
    with bp_cache.attach(model), torch.no_grad():
        out_bp = model.generate(
            input_ids=input_ids,
            cache=bp_cache,
            use_cache=True,
            max_new_tokens=gen_steps,
            do_sample=False,
        )
    bp_kv_bytes = torch.cuda.memory_allocated(device) - base_mem_bp
    assert out_bp.shape == (bsz, l_prefill + gen_steps)
    # generate 先执行 1 次 Prefill 得出首 token，再执行 gen_steps - 1 次单步 Decode
    assert len(bp_cache.read_bits_samples) == cfg.num_hidden_layers * (gen_steps - 1)

    # 验证各层均无长期驻留 full keys/values，且尾窗有界
    for lyr in bp_cache.layers:
        assert isinstance(lyr, BitPlaneCacheLayer)
        assert lyr.keys is None and lyr.values is None
        assert lyr.pool is not None
        tail_l = lyr.pool.sequences[lyr.seq_ids[0]].tail_len
        assert 32 <= tail_l < 64
        # generate() 生成最后一个 token 后不再作为输入喂回模型，故 cache 总长为 l_prefill + gen_steps - 1
        assert lyr.get_seq_length() == l_prefill + gen_steps - 1

    # 验证真实 CUDA allocator 净增 KV 显存相对 DynamicCache 压缩超过 2.5 倍
    real_vram_ratio = nat_kv_bytes / max(bp_kv_bytes, 1)
    assert real_vram_ratio > 2.5, (
        f"BitPlaneDynamicCache 真机显存压缩不足: native={nat_kv_bytes/1024**2:.2f} MB vs bitplane={bp_kv_bytes/1024**2:.2f} MB"
    )
    assert bp_cache.memory_stats()["compression_ratio"] > 2.8


def test_bitplane_dynamic_cache_deepcopy_safe() -> None:
    """验证在 attach(model) 内对 BitPlaneDynamicCache 执行 copy.deepcopy 后多步自回归依然与直通版逐位一致。"""
    torch.manual_seed(7450)
    device = torch.device("cuda:0")
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    pref_ids = torch.randint(0, 200, (1, 160), device=device, dtype=torch.long)
    step_ids = torch.randint(0, 200, (1, 40), device=device, dtype=torch.long)

    cache_orig = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32, hot_ratio=0.15, warm_ratio=0.35)
    with cache_orig.attach(model), torch.no_grad():
        model(input_ids=pref_ids, cache=cache_orig, use_cache=True)
        cache_copy = copy.deepcopy(cache_orig)

        logits_a = []
        for t in range(40):
            o = model(input_ids=step_ids[:, t : t + 1], cache=cache_orig, use_cache=True)
            logits_a.append(o.logits[:, -1, :].float())

        logits_b = []
        for t in range(40):
            o = model(input_ids=step_ids[:, t : t + 1], cache=cache_copy, use_cache=True)
            logits_b.append(o.logits[:, -1, :].float())

    max_diff = float((torch.stack(logits_a) - torch.stack(logits_b)).abs().max().item())
    assert max_diff == 0.0, f"deepcopy 后自回归输出必须逐位一致，实际误差={max_diff}"
    assert cache_copy.layers[0].get_seq_length() == 200


def _find_cached_model_dir(model_name: str) -> Path | None:
    env_root = os.environ.get("BITPLANE_MODEL_CACHE")
    if env_root:
        p = Path(env_root) / model_name
        if p.exists():
            return p
    local_p = Path(__file__).resolve().parents[1] / ".model_cache" / model_name
    return local_p if local_p.exists() else None


def test_real_qwen_bitplane_dynamic_cache_optional() -> None:
    """若本地存在 Qwen3-0.6B 与 Qwen3.5-0.8B 缓存则验证真模型 BitPlaneDynamicCache.generate 全流程与显存压缩。"""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    m_dir = _find_cached_model_dir("Qwen3-0.6B")
    if m_dir is None:
        pytest.skip("未在 BITPLANE_MODEL_CACHE 或 .model_cache 找到本地 Qwen3-0.6B 权重")

    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(str(m_dir))
    model = AutoModelForCausalLM.from_pretrained(str(m_dir), dtype=torch.bfloat16).to(device).eval()
    prompt_ids = tok.encode("PagedAttention maps logical KV blocks onto a physical pool. " * 40, return_tensors="pt").to(device)

    bp_cache = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32, hot_ratio=0.15, warm_ratio=0.35)
    with bp_cache.attach(model), torch.no_grad():
        gen_out = model.generate(input_ids=prompt_ids, cache=bp_cache, max_new_tokens=40, do_sample=False)
    assert gen_out.shape[1] == prompt_ids.shape[1] + 40
    st = bp_cache.memory_stats()
    assert st["num_paged_layers"] == 28
    assert st["compression_ratio"] > 2.5
    del model, bp_cache
    gc.collect()
    torch.cuda.empty_cache()

    m35_dir = _find_cached_model_dir("Qwen3.5-0.8B")
    if m35_dir is not None:
        tok35 = AutoTokenizer.from_pretrained(str(m35_dir))
        model35 = Qwen3_5ForConditionalGeneration.from_pretrained(str(m35_dir), dtype=torch.bfloat16).to(device).eval()
        p35 = tok35.encode("Hybrid GatedDeltaNet plus scarce global softmax attention. " * 40, return_tensors="pt").to(device)
        bp35 = BitPlaneDynamicCache(config=model35.config, page_size=32, recent_tokens=32, hot_ratio=0.15, warm_ratio=0.35)
        with bp35.attach(model35), torch.no_grad():
            g35 = model35.generate(input_ids=p35, cache=bp35, max_new_tokens=40, do_sample=False)
        assert g35.shape[1] == p35.shape[1] + 40
        st35 = bp35.memory_stats()
        assert st35["num_paged_layers"] == 6
        assert st35["compression_ratio"] > 2.5
        del model35, bp35
        gc.collect()
        torch.cuda.empty_cache()


def test_prefix_cache_invalidation_on_single_ref_inplace_write() -> None:
    """P0 回归测试：单引用 (`ref_count == 1`) 原地修改物理页时必须解除旧哈希绑定，禁止新序列读到被篡改的脏数据。"""
    torch.manual_seed(7501)
    h_kv, d = 4, 128
    k_orig = torch.randn((1, h_kv, 64, d), device="cuda:0", dtype=torch.float16)
    v_orig = torch.randn((1, h_kv, 64, d), device="cuda:0", dtype=torch.float16)

    pool = PagedBitPlaneKVPool(num_kv_heads=h_kv, head_dim=d, page_size=32, capacity=8, kv_dtype=torch.float16, device="cuda:0")
    seq_a = pool.allocate_sequence()
    pk_h = b"prefix-page-0-hash"
    pool.append_kv(seq_a, k_orig, v_orig, recent_tokens=32, prefix_keys=[pk_h])

    blk_p = pool.sequences[seq_a].block_ids[0]
    assert pool.ref_counts[blk_p] == 1
    assert pool.prefix_table.get(pk_h) == blk_p

    # 对唯一引用的第 0 页执行 write_page 原地覆写新内容
    k_mut = torch.randn((1, h_kv, 32, d), device="cuda:0", dtype=torch.float16) * 4.0
    v_mut = torch.randn((1, h_kv, 32, d), device="cuda:0", dtype=torch.float16) * 4.0
    blk_after = pool.write_page(seq_a, 0, k_mut, v_mut)
    assert blk_after == blk_p
    assert pool.ref_counts[blk_p] == 1
    # 原哈希 H 必须已失效，不能再命中已被修改的物理块 P
    assert pk_h not in pool.prefix_table
    assert blk_p not in pool._block_prefix_key

    # 新建序列 B 携带旧哈希 H 与原 KV 追加，不得命中 seq_a 的脏物理块，且各自解码/解包数据完全独立正确
    seq_b = pool.allocate_sequence()
    pool.append_kv(seq_b, k_orig, v_orig, recent_tokens=32, prefix_keys=[pk_h])
    blk_b = pool.sequences[seq_b].block_ids[0]
    assert blk_b != blk_p
    assert pool.prefix_table.get(pk_h) == blk_b
    assert not torch.equal(pool.unpack_sequence(seq_a)["k_int4"], pool.unpack_sequence(seq_b)["k_int4"])

    # 验证直接使用 copy_on_write(ref==1) 也会预先解绑前缀哈希
    blk_b_cow = pool.copy_on_write(seq_b, 0)
    assert blk_b_cow == blk_b
    assert pk_h not in pool.prefix_table

    pool.free_sequence(seq_a)
    pool.free_sequence(seq_b)
    assert len(pool.free_blocks) == pool.capacity
    assert len(pool.prefix_table) == 0


def test_hf_left_padding_batch_attention_mask_correctness() -> None:
    """P0 回归测试：验证不等长 Left-Padding Batch 下 Decode Mask 语义与 strict_mask=True 显式报错。"""
    torch.manual_seed(7502)
    device = torch.device("cuda:0")
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()

    # 序列 0 有效长度 96（左侧填充 64 个 token 0），序列 1 有效长度 160
    s0_valid = torch.randint(10, 200, (1, 96), device=device, dtype=torch.long)
    s1_valid = torch.randint(10, 200, (1, 160), device=device, dtype=torch.long)
    pad_left = torch.zeros((1, 64), device=device, dtype=torch.long)
    s0_padded = torch.cat([pad_left, s0_valid], dim=1)
    batch_ids = torch.cat([s0_padded, s1_valid], dim=0)
    batch_mask = torch.cat(
        [
            torch.cat([torch.zeros((1, 64), device=device, dtype=torch.long), torch.ones((1, 96), device=device, dtype=torch.long)], dim=1),
            torch.ones((1, 160), device=device, dtype=torch.long),
        ],
        dim=0,
    )
    next_tok = torch.randint(10, 200, (2, 1), device=device, dtype=torch.long)

    # 1) 单样本无 padding 对照（同模式 BitPlane triton_int4 与 Native BF16）
    nat_s0 = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=s0_valid, cache=nat_s0, use_cache=True)
        out_nat_s0 = model(input_ids=next_tok[0:1], cache=nat_s0, use_cache=True).logits[0, -1, :].float()

    bp_i4_single = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32, mode="triton_int4")
    with bp_i4_single.attach(model), torch.no_grad():
        model(input_ids=s0_valid, cache=bp_i4_single, use_cache=True)
        out_i4_s0_single = model(input_ids=next_tok[0:1], cache=bp_i4_single, use_cache=True).logits[0, -1, :].float()

    # 2) Batch Left-padding + BitPlane triton_int4：验证掩码被真实应用，且序列 0 结果精确一致于无 padding 单样本推理
    bp_i4_batch = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32, mode="triton_int4")
    pos_pref = batch_mask.long().cumsum(-1) - 1
    pos_pref.masked_fill_(batch_mask == 0, 0)
    pos_dec = batch_mask.sum(dim=-1, keepdim=True).long()
    full_dec_mask = torch.cat([batch_mask, torch.ones((2, 1), device=device, dtype=torch.long)], dim=1)

    with bp_i4_batch.attach(model), torch.no_grad():
        model(input_ids=batch_ids, attention_mask=batch_mask, position_ids=pos_pref, cache=bp_i4_batch, use_cache=True)
        out_bp_i4 = model(
            input_ids=next_tok,
            attention_mask=full_dec_mask,
            position_ids=pos_dec,
            cache=bp_i4_batch,
            use_cache=True,
        )
    diff_i4_exact = float((out_bp_i4.logits[0, -1, :].float() - out_i4_s0_single).abs().max().item())
    diff_i4_nat = float((out_bp_i4.logits[0, -1, :].float() - out_nat_s0).abs().max().item())
    assert diff_i4_exact < 1.5e-2, f"Left-padding Batch vs 单样本同模式差异过大（疑忽略 Mask）: {diff_i4_exact}"
    assert diff_i4_nat < 5e-2, f"Left-padding Batch vs Native BF16 差异过大: {diff_i4_nat}"

    # 3) Batch Left-padding + BitPlane triton_dualstream：无 NaN，且严格对齐原生 Left-padding DynamicCache
    nat_batch = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=batch_ids, attention_mask=batch_mask, position_ids=pos_pref, cache=nat_batch, use_cache=True)
        out_nat_batch = model(
            input_ids=next_tok, attention_mask=full_dec_mask, position_ids=pos_dec, cache=nat_batch, use_cache=True
        )
    bp_ds_batch = BitPlaneDynamicCache(
        config=model.config, page_size=32, recent_tokens=32, mode="triton_dualstream", hot_ratio=0.5, warm_ratio=0.5
    )
    with bp_ds_batch.attach(model), torch.no_grad():
        model(input_ids=batch_ids, attention_mask=batch_mask, position_ids=pos_pref, cache=bp_ds_batch, use_cache=True)
        out_bp_ds = model(
            input_ids=next_tok, attention_mask=full_dec_mask, position_ids=pos_dec, cache=bp_ds_batch, use_cache=True
        )
    assert not torch.isnan(out_bp_ds.logits).any()
    diff_ds_s0 = float((out_bp_ds.logits[0, -1, :].float() - out_nat_batch.logits[0, -1, :].float()).abs().max().item())
    assert diff_ds_s0 < 3e-2, f"Left-padding Batch DualStream 误差过大: {diff_ds_s0}"

    # 4) strict_mask=True 遇到非平凡 Mask 时必须显式报错
    bp_strict = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32, strict_mask=True)
    with bp_strict.attach(model), torch.no_grad():
        model(input_ids=batch_ids, attention_mask=batch_mask, position_ids=pos_pref, cache=bp_strict, use_cache=True)
        with pytest.raises(NotImplementedError):
            model(input_ids=next_tok, attention_mask=full_dec_mask, position_ids=pos_dec, cache=bp_strict, use_cache=True)


def test_hf_chunked_prefill_matches_single_prefill() -> None:
    """回归测试：验证分块 Chunked Prefill (logits_to_keep=1) 后解码输出与单次 Prefill 一致。"""
    torch.manual_seed(7503)
    device = torch.device("cuda:0")
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    pref_ids = torch.randint(1, 200, (1, 256), device=device, dtype=torch.long)
    dec_tok = torch.randint(1, 200, (1, 1), device=device, dtype=torch.long)

    c_full = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32, hot_ratio=0.25, warm_ratio=0.5)
    with c_full.attach(model), torch.no_grad():
        model(input_ids=pref_ids, cache=c_full, use_cache=True, logits_to_keep=1)
        out_full = model(input_ids=dec_tok, cache=c_full, use_cache=True).logits[:, -1, :].float()

    c_chunk = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32, hot_ratio=0.25, warm_ratio=0.5)
    with c_chunk.attach(model), torch.no_grad():
        model(input_ids=pref_ids[:, :128], cache=c_chunk, use_cache=True, logits_to_keep=1)
        model(input_ids=pref_ids[:, 128:], cache=c_chunk, use_cache=True, logits_to_keep=1)
        out_chunk = model(input_ids=dec_tok, cache=c_chunk, use_cache=True).logits[:, -1, :].float()

    assert c_chunk.get_seq_length() == 257
    diff = (out_full - out_chunk).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    assert mean_diff < 2e-2 and max_diff < 8e-2, (
        f"Chunked Prefill 后 Decode 输出偏差过大: mean={mean_diff}, max={max_diff}"
    )
    assert out_full.argmax(dim=-1).item() == out_chunk.argmax(dim=-1).item()


def test_concurrent_nested_attach_and_activate_safety() -> None:
    """P1 回归测试：验证引用计数 attach() 乱序退出不破坏模型配置，以及无状态 activate() 正常工作。"""
    torch.manual_seed(7504)
    device = torch.device("cuda:0")
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    orig_impl = model.config._attn_implementation

    c1 = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32)
    c2 = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32)

    cm1 = c1.attach(model)
    cm2 = c2.attach(model)
    cm1.__enter__()
    assert model.config._attn_implementation == "bitplane_paged"
    cm2.__enter__()
    assert model.config._attn_implementation == "bitplane_paged"
    # 故意先退出外层 cm1，再退出内层 cm2（模拟并发异步请求乱序释放）
    cm1.__exit__(None, None, None)
    assert model.config._attn_implementation == "bitplane_paged"
    cm2.__exit__(None, None, None)
    assert model.config._attn_implementation == orig_impl

    # 验证真实多线程交错 attach 不会造成状态损毁
    import threading
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _worker(tid: int) -> None:
        try:
            ct = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32)
            with ct.attach(model):
                assert model.config._attn_implementation == "bitplane_paged"
                barrier.wait(timeout=5.0)
                if tid == 0:
                    # 线程 0 稍晚退出也不破坏线程 1
                    pass
        except BaseException as exc:
            errors.append(exc)

    t0 = threading.Thread(target=_worker, args=(0,))
    t1 = threading.Thread(target=_worker, args=(1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    assert not errors, f"多线程 attach 异常: {errors}"
    assert model.config._attn_implementation == orig_impl

    # 验证无状态 activate() 配合预置 attn_implementation="bitplane_paged"
    model.set_attn_implementation("bitplane_paged")
    pref = torch.randint(1, 200, (1, 128), device=device, dtype=torch.long)
    dec = torch.randint(1, 200, (1, 1), device=device, dtype=torch.long)
    c_act = BitPlaneDynamicCache(config=model.config, page_size=32, recent_tokens=32, collect_stats=True)
    with c_act.activate(), torch.no_grad():
        model(input_ids=pref, cache=c_act, use_cache=True)
        model(input_ids=dec, cache=c_act, use_cache=True)
    assert len(c_act.read_bits_samples) == 2
    assert model.config._attn_implementation == "bitplane_paged"
    model.set_attn_implementation(orig_impl)
