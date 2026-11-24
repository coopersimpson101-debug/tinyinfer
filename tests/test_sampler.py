"""Sampling: greedy, temperature, top-k, top-p, repetition penalty."""
import ctypes as C

import numpy as np
from conftest import SamplerParams, fptr, iptr


def params(temp=1.0, top_k=0, top_p=1.0, rep=1.0, last_n=0, seed=1):
    return SamplerParams(temp, top_k, top_p, rep, last_n, seed)


def sample(lib, logits, p, history=(), rng=None):
    x = np.array(logits, dtype=np.float32)
    h = np.array(history, dtype=np.int32)
    st = rng if rng is not None else C.c_uint64(p.seed)
    return lib.ti_sample(fptr(x), len(x), C.byref(p), iptr(h) if len(h) else None, len(h), C.byref(st))


def draw(lib, logits, p, n=20000):
    st = C.c_uint64(p.seed)
    counts = np.zeros(len(logits))
    for _ in range(n):
        counts[sample(lib, logits, p, rng=st)] += 1
    return counts / n


LOGITS = [1.0, 3.0, 2.5, -1.0, 0.0, 2.9]


def test_greedy(lib):
    assert sample(lib, LOGITS, params(temp=0.0)) == 1
    assert sample(lib, LOGITS, params(temp=0.8, top_k=1)) == 1
    assert sample(lib, LOGITS, params(temp=0.8, top_p=1e-6)) == 1


def test_temperature_distribution(lib):
    for temp in (1.0, 0.5):
        freq = draw(lib, LOGITS, params(temp=temp))
        z = np.exp(np.array(LOGITS) / temp)
        p = z / z.sum()
        assert np.abs(freq - p).max() < 0.015, (temp, freq, p)


def test_top_k(lib):
    freq = draw(lib, LOGITS, params(top_k=2), n=5000)
    assert set(np.nonzero(freq)[0]) == {1, 5}
    z = np.exp([3.0, 2.9])
    assert abs(freq[1] - z[0] / z.sum()) < 0.03


def test_top_p(lib):
    p = np.exp(LOGITS) / np.exp(LOGITS).sum()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    keep = set(order[: int(np.searchsorted(cum, 0.7) + 1)].tolist())
    freq = draw(lib, LOGITS, params(top_p=0.7), n=5000)
    assert set(np.nonzero(freq)[0]) == keep
    # with a large vocabulary the pre-filter path is used; result must stay inside the nucleus
    big = np.full(50000, -10.0, dtype=np.float32)
    big[[10, 20, 30]] = [5.0, 4.8, 4.6]
    st = C.c_uint64(3)
    for _ in range(200):
        assert sample(lib, big, params(top_p=0.9), rng=st) in (10, 20, 30)


def test_repetition_penalty(lib):
    # token 1 (logit 3.0) repeated -> 3.0/1.5 = 2.0 < 2.9 so token 5 wins greedily
    assert sample(lib, LOGITS, params(temp=0.0, rep=1.5, last_n=8), history=[1, 1, 4]) == 5
    # outside the window: no penalty
    assert sample(lib, LOGITS, params(temp=0.0, rep=1.5, last_n=1), history=[1, 4]) == 1
    # negative logits are multiplied (pushed further down)
    x = np.array([-1.0, -1.5], dtype=np.float32)
    p = params(temp=0.0, rep=2.0, last_n=4)
    h = np.array([0], dtype=np.int32)
    st = C.c_uint64(1)
    assert lib.ti_sample(fptr(x), 2, C.byref(p), iptr(h), 1, C.byref(st)) == 1
    assert x[0] == -2.0


def test_seed_determinism(lib):
    a = [sample(lib, LOGITS, params(seed=99), rng=C.c_uint64(99)) for _ in range(5)]
    st1, st2 = C.c_uint64(7), C.c_uint64(7)
    s1 = [sample(lib, LOGITS, params(), rng=st1) for _ in range(50)]
    s2 = [sample(lib, LOGITS, params(), rng=st2) for _ in range(50)]
    assert s1 == s2 and len(set(s1)) > 1 and len(a) == 5


def test_defaults(lib):
    d = lib.ti_sampler_default()
    assert d.top_k == 40 and abs(d.top_p - 0.9) < 1e-6 and abs(d.repeat_penalty - 1.1) < 1e-6
