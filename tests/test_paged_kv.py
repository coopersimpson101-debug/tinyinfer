"""Paged KV cache vs the contiguous cache: logits must be bit-identical."""
import numpy as np
import pytest

TOKENS = [5, 77, 300, 12, 9, 41, 2, 150, 33, 7, 88, 19, 240, 64, 3, 101, 55, 6, 18, 200]


def incremental(m, toks, seq=0, prefill=7):
    """Prefill `prefill` tokens, then decode the rest one at a time; logits per step."""
    out = [m.forward_seq(seq, toks[:prefill], 0, all_logits=True)]
    for i in range(prefill, len(toks)):
        out.append(m.forward_seq(seq, toks[i:i + 1], i, all_logits=True))
    return np.concatenate(out)


@pytest.mark.parametrize("fmt", ["f32", "q8_0", "q4_0"])
@pytest.mark.parametrize("block", [1, 3, 16])
def test_paged_matches_contiguous_bitwise(models, open_model, fmt, block):
    ref = incremental(open_model(models[fmt], threads=2), TOKENS)
    pm = open_model(models[fmt], threads=2)
    pm.enable_paged(block)
    got = incremental(pm, TOKENS)
    assert np.array_equal(ref, got), np.abs(ref - got).max()


def test_paged_batched_prefill_across_blocks(models, open_model):
    m0 = open_model(models["f32_untied"], threads=2)
    m1 = open_model(models["f32_untied"], threads=2)
    m1.enable_paged(4)
    assert np.array_equal(m0.forward(TOKENS), m1.forward(TOKENS))


def test_multiple_sequences_interleaved(models, open_model, lib):
    a, b = TOKENS[:12], TOKENS[::-1][:12]
    ref_a = incremental(open_model(models["f32"], threads=2), a, prefill=3)
    ref_b = incremental(open_model(models["f32"], threads=2), b, prefill=5)
    m = open_model(models["f32"], threads=2)
    m.enable_paged(4, 16)
    out_a = [m.forward_seq(0, a[:3], 0)]
    out_b = [m.forward_seq(1, b[:5], 0)]
    for i in range(3, 12):  # interleave the two sequences step by step
        out_a.append(m.forward_seq(0, a[i:i + 1], i))
        j = i + 2
        if j < 12:
            out_b.append(m.forward_seq(1, b[j:j + 1], j))
    assert np.array_equal(np.concatenate(out_a), ref_a)
    assert np.array_equal(np.concatenate(out_b), ref_b)
    assert lib.ti_kv_seq_len(m.h, 0) == 12 and lib.ti_kv_seq_len(m.h, 1) == 12
    assert lib.ti_kv_blocks_free(m.h) == 16 - 6


def test_rollback_frees_blocks_and_recomputes(models, open_model, lib):
    m = open_model(models["f32"], threads=2)
    m.enable_paged(4, 8)
    m.forward_seq(0, TOKENS[:16], 0)
    assert lib.ti_kv_blocks_free(m.h) == 4
    # rewrite from position 10 with different tokens: positions >= 10 are discarded
    alt = TOKENS[:10] + [1, 2, 3]
    got = m.forward_seq(0, alt[10:], 10)
    assert lib.ti_kv_seq_len(m.h, 0) == 13 and lib.ti_kv_blocks_free(m.h) == 4
    ref = open_model(models["f32"], threads=2).forward(alt)[10:]
    assert np.array_equal(got, ref)
    assert lib.ti_kv_seq_trim(m.h, 0, 0) == 0 and lib.ti_kv_blocks_free(m.h) == 8


def test_fork_shares_prefix_with_copy_on_write(models, open_model, lib):
    m = open_model(models["f32"], threads=2)
    m.enable_paged(4, 16)
    prefix = TOKENS[:6]  # 1.5 blocks: the second block is shared *and* partially filled
    m.forward_seq(0, prefix, 0)
    used = 16 - lib.ti_kv_blocks_free(m.h)
    assert lib.ti_kv_seq_copy(m.h, 1, 0) == 0
    assert 16 - lib.ti_kv_blocks_free(m.h) == used  # fork costs no blocks
    ga = m.forward_seq(0, [11, 12, 13], 6)
    gb = m.forward_seq(1, [21, 22], 6)  # writes into the shared block -> copied first
    ref = open_model(models["f32"], threads=2)
    assert np.array_equal(ga, ref.forward(prefix + [11, 12, 13])[6:])
    ref2 = open_model(models["f32"], threads=2)
    assert np.array_equal(gb, ref2.forward(prefix + [21, 22])[6:])


def test_pool_exhaustion_is_an_error(models, open_model, lib):
    m = open_model(models["f32"], threads=2)
    m.enable_paged(4, 2)  # 8 positions in total
    m.forward_seq(0, TOKENS[:8], 0)
    with pytest.raises(RuntimeError, match="exhausted"):
        m.forward_seq(0, TOKENS[8:9], 8)
    assert lib.ti_kv_seq_len(m.h, 0) == 8  # failed call left the sequence intact
    with pytest.raises(RuntimeError, match="cannot write"):
        m.forward_seq(1, TOKENS[:1], 3)  # gap in a sequence


def test_contiguous_rejects_other_sequences(models, open_model):
    m = open_model(models["f32"], threads=2)
    with pytest.raises(RuntimeError, match="paged"):
        m.forward_seq(1, TOKENS[:2], 0)
