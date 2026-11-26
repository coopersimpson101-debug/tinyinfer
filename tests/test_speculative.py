"""Speculative decoding: the acceptance rule, greedy exactness and distribution checks."""
import ctypes as C
import re
import subprocess

import numpy as np
import pytest
from conftest import BUILD, EXE, SamplerParams, fptr

CLI = BUILD / f"tinyinfer{EXE}"


def params(temp=0.0, top_k=0, top_p=1.0, seed=1):
    return SamplerParams(temperature=temp, top_k=top_k, top_p=top_p, repeat_penalty=1.0,
                         repeat_last_n=0, seed=seed)


@pytest.fixture(scope="module")
def draft_model(tmp_path_factory):
    """A different (1-layer, other seed) random model with the same tokenizer."""
    from make_tiny_model import build
    from tinyinfer_ref import quant
    d = tmp_path_factory.mktemp("draft")
    return build(str(d / "draft.gguf"), quant.F32, seed=7, n_layers=1)


def probs(lib, logits, p):
    x = np.ascontiguousarray(logits, dtype=np.float32)
    out = np.zeros_like(x)
    lib.ti_sampler_probs(fptr(x), len(x), C.byref(p), fptr(out))
    return out


def np_probs(logits, temp, top_k, top_p):
    z = logits.astype(np.float64) / temp
    order = np.argsort(-z, kind="stable")
    keep = order[:top_k] if top_k else order
    e = np.exp(z[keep] - z[keep].max())
    pk = e / e.sum()
    if top_p < 1:
        c = np.cumsum(pk)
        n = int(np.searchsorted(c, top_p) + 1)
        keep, pk = keep[:n], pk[:n] / pk[:n].sum()
    out = np.zeros(len(z))
    out[keep] = pk
    return out


def test_sampler_probs_matches_numpy(lib):
    rng = np.random.default_rng(0)
    for temp, k, p in [(1.0, 0, 1.0), (0.7, 10, 1.0), (1.3, 0, 0.9), (0.8, 40, 0.95)]:
        lg = rng.standard_normal(500).astype(np.float32) * 3
        got = probs(lib, lg, params(temp, k, p))
        assert abs(got.sum() - 1) < 1e-5
        assert np.abs(got - np_probs(lg, temp, k, p)).max() < 1e-6
    g = probs(lib, lg, params(0.0))
    assert g.sum() == 1 and g.argmax() == lg.argmax()


@pytest.mark.parametrize("case", ["random", "q_misses_support", "identical"])
def test_acceptance_rule_reproduces_target(lib, case):
    """Draw x ~ q, apply the rule: the emitted token must be distributed as p."""
    rng = np.random.default_rng(1)
    V = 6
    p = rng.dirichlet(np.ones(V)).astype(np.float32)
    q = rng.dirichlet(np.ones(V)).astype(np.float32)
    if case == "q_misses_support":
        q[0] = 0
        q /= q.sum()
    if case == "identical":
        q = p.copy()
    scratch = np.zeros(V, np.float32)
    state = C.c_uint64(12345)
    acc = C.c_int(0)
    N = 200_000
    counts = np.zeros(V)
    n_acc = 0
    qd = np.cumsum(q.astype(np.float64))
    draws = np.minimum(np.searchsorted(qd, rng.random(N) * qd[-1], side="right"), V - 1)
    for x in draws:
        t = lib.ti_spec_accept(fptr(p), fptr(q), V, int(x), fptr(scratch), C.byref(state),
                               C.byref(acc))
        counts[t] += 1
        n_acc += acc.value
    emp = counts / N
    assert 0.5 * np.abs(emp - p).sum() < 0.006, (emp, p)
    # acceptance probability = sum_x min(p(x), q(x))
    assert abs(n_acc / N - np.minimum(p, q).sum()) < 0.006


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("k", [1, 3, 6])
def test_greedy_speculative_equals_greedy_target(models, open_model, draft_model, paged, k):
    tgt = open_model(models["f32"], threads=2)
    base, _ = tgt.generate([5, 77, 300, 12], 40, params(0.0))
    t2 = open_model(models["f32"], threads=2)
    d = open_model(draft_model, threads=2)
    if paged:
        t2.enable_paged(8)
        d.enable_paged(8)
    spec, st = t2.generate([5, 77, 300, 12], 40, params(0.0), draft=d, k=k)
    assert spec == base
    assert st.generated == 40 and st.rounds < 40 + 1
    assert 0 <= st.accepted <= st.drafted


def test_draft_equal_to_target_accepts_everything(models, open_model):
    t = open_model(models["q8_0"], threads=2)
    d = open_model(models["q8_0"], threads=2)
    out, st = t.generate([5, 77], 30, params(0.9, 0, 1.0, seed=3), draft=d, k=4)
    assert len(out) == 30
    assert st.accepted == st.drafted  # p == q at every position (up to rounding)
    assert st.rounds == 6             # 5 tokens per target call: 30 / 5


def test_greedy_speculative_quantised_target(models, open_model, draft_model):
    t = open_model(models["q4_0"], threads=2)
    base, _ = t.generate([9, 41, 2], 32, params(0.0))
    t2 = open_model(models["q4_0"], threads=2)
    spec, _ = t2.generate([9, 41, 2], 32, params(0.0), draft=open_model(draft_model), k=4)
    assert spec == base


def test_sampled_first_token_has_target_distribution(models, open_model, draft_model, lib):
    """End to end at temperature 1.5 / top-k 8: the first emitted token goes through the
    draft + accept/residual path; its empirical law must match the target's."""
    tgt = open_model(models["f32"], threads=1)
    d = open_model(draft_model, threads=1)
    prompt = [5, 77, 300]
    sp = params(1.5, 8, 1.0)
    p_exact = probs(lib, tgt.forward(prompt)[-1], sp)
    N = 3000
    counts = np.zeros(len(p_exact))
    acc = drafted = 0
    for s in range(N):
        sp.seed = 1000 + s
        out, st = tgt.generate(prompt, 3, sp, draft=d, k=2)
        counts[out[0]] += 1
        acc += st.accepted
        drafted += st.drafted
    emp = counts / N
    assert 0 < acc < drafted  # rejections do happen with this draft
    assert 0.5 * np.abs(emp - p_exact).sum() < 0.04


def test_cli_speculative_flags(models, draft_model):
    if not CLI.exists():
        pytest.skip("CLI not built")
    common = [CLI, "-m", models["f32"], "-p", "Once upon a time", "-n", "24", "--temp", "0",
              "--repeat-penalty", "1", "-t", "2", "--ignore-eos"]
    run = lambda extra: subprocess.run([str(a) for a in common + extra], capture_output=True,  # noqa: E731
                                       timeout=120)
    plain = run([])
    spec = run(["--draft", draft_model, "--spec-k", "3"])
    paged = run(["--draft", draft_model, "--spec-k", "3", "--kv-paged", "--kv-block", "8"])
    base = run(["--spec-baseline"])
    for r in (plain, spec, paged, base):
        assert r.returncode == 0, r.stderr.decode()
    assert spec.stdout == plain.stdout == paged.stdout == base.stdout
    err = spec.stderr.decode()
    m = re.search(r"acceptance=([\d.]+)% tokens/round=([\d.]+)", err)
    assert m and 0 <= float(m.group(1)) <= 100
    assert re.search(r"decode:  24 tokens in [\d.]+ ms \([\d.]+ tok/s\)", err)
    assert "[paged KV]" in paged.stderr.decode()
