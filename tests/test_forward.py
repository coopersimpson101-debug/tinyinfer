"""Full forward pass: C engine vs the NumPy reference on tiny random models."""
import numpy as np
import pytest
from tinyinfer_ref.model import QwenRef

TOKENS = [5, 77, 300, 12, 9, 401, 33, 250, 2, 101, 88, 7]

# max |C - ref| / std(ref logits). Quantised models use the activation-quantising
# reference, so the remaining gap is float rounding plus rare Q8_0 rounding flips.
TOL = {"f32": 1e-5, "f16": 1e-5, "q8_0": 2e-2, "q4_0": 2e-2,
       "f32_untied": 1e-5, "q8_0_untied": 2e-2, "f32_mha": 1e-5}


def rel_err(a, b):
    return float(np.abs(a - b).max() / b.std())


@pytest.mark.parametrize("name", list(TOL))
def test_forward_logits_match_reference(open_model, models, name):
    m = open_model(models[name])
    ref = QwenRef(models[name]).forward(TOKENS)
    got = m.forward(TOKENS, all_logits=True)
    assert got.shape == ref.shape
    assert rel_err(got, ref) < TOL[name]
    assert np.array_equal(got.argmax(-1), ref.argmax(-1)) or name.startswith("q")


def test_config_parsed(open_model, models):
    m = open_model(models["f32_untied"])
    c = m.cfg
    assert (c.n_layers, c.dim, c.ffn_dim, c.n_heads, c.n_kv_heads, c.head_dim) == (3, 64, 128, 4, 2, 16)
    assert c.tied_embeddings == 0 and c.has_qkv_bias == 1
    assert abs(c.rope_theta - 10000.0) < 1e-3
    assert open_model(models["f32"]).cfg.tied_embeddings == 1


@pytest.mark.parametrize("name", ["f32", "q4_0"])
def test_kv_cache_incremental_equals_batch(open_model, models, name):
    m = open_model(models[name])
    full = m.forward(TOKENS, all_logits=True)
    m2 = open_model(models[name])
    m2.forward(TOKENS[:5], pos=0, all_logits=False)
    rows = []
    for i in range(5, len(TOKENS)):
        rows.append(m2.forward([TOKENS[i]], pos=i, all_logits=False)[0])
    inc = np.stack(rows)
    tol = 1e-5 if name == "f32" else 2e-2
    assert rel_err(inc, full[5:]) < tol


def test_last_logits_only(open_model, models):
    m = open_model(models["f32"])
    full = m.forward(TOKENS, all_logits=True)
    last = m.forward(TOKENS, all_logits=False)
    np.testing.assert_allclose(last[0], full[-1], rtol=1e-5, atol=1e-5)


def test_long_sequence_crosses_batch_chunks(open_model, models):
    """> 64 tokens forces several prefill chunks (TI_MAX_BATCH) through the KV cache."""
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 500, 150).tolist()
    m = open_model(models["f32"])
    got = m.forward(toks, all_logits=True)
    ref = QwenRef(models["f32"]).forward(toks)
    assert rel_err(got, ref) < 1e-5


def test_context_overflow_is_an_error(open_model, models):
    m = open_model(models["f32"], ctx=8)
    with pytest.raises(RuntimeError, match="context overflow"):
        m.forward(list(range(9)))


def test_bad_token_is_an_error(open_model, models):
    m = open_model(models["f32"])
    with pytest.raises(RuntimeError, match="out of range"):
        m.forward([m.cfg.vocab_size + 5])


def test_load_errors(lib, tmp_path):
    from conftest import Model
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(RuntimeError, match="not a GGUF"):
        Model(lib, bad)
    with pytest.raises(RuntimeError, match="cannot open"):
        Model(lib, tmp_path / "missing.gguf")


def test_quantised_models_track_float_model(open_model, models):
    """Same seed -> same float weights; measures what quantisation does to the logits.

    Relative L2 error on this dim=64 toy (measured: f16 ~0.1%, q8_0 ~3%, q4_0 ~29%);
    real models with dim >= 1536 have far more averaging per dot product."""
    f = open_model(models["f32"]).forward(TOKENS)
    for name, tol in (("f16", 5e-3), ("q8_0", 0.06), ("q4_0", 0.5)):
        q = open_model(models[name]).forward(TOKENS)
        assert np.linalg.norm(q - f) / np.linalg.norm(f) < tol, name
        if name != "q4_0":
            assert (q.argmax(-1) == f.argmax(-1)).mean() >= 0.75, name
