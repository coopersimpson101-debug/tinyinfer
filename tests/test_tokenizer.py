"""Tokenizer: pre-tokeniser vs regex, C vs Python BPE, encode/decode roundtrip."""
import ctypes as C
import random

import numpy as np
import pytest
from tinyinfer_ref import tokenizer as T
from tinyinfer_ref.gguf import read_gguf

SAMPLES = [
    "Hello world",
    "Once upon a time, there was a tiny inference engine.",
    "  leading spaces and trailing   ",
    "tabs\tand\nnewlines\r\n\r\nwindows\n\n\n",
    "it's we'll they're I'M you've he'd she'S",
    "numbers 1234567890 and 3.14159, 2026-11-24",
    "code: for (int i = 0; i < n; i++) { y[i] += a * x[i]; }",
    "def f(x):\n    return x @ w.T  # comment",
    "symbols !!! ??? ... --- *** ### $$$ %%% ^^^",
    "Café naïve résumé über straße",
    "добрый день, καλημέρα",
    "你好，世界。今天天气很好。",
    "こんにちは世界 한국어",
    "emoji \U0001F600\U0001F680 mixed\U0001F600text",
    "fullwidth ＡＢＣ１２３！",
    "a nbsp and　ideographic space",
    "'quoted' \"double\" `back`",
    "",
    " ",
    "\n",
    "x",
]


def c_pretok(lib, text):
    b = text.encode("utf-8")
    ends = (C.c_int * (len(b) + 1))()
    n = lib.ti_pretokenize(b, len(b), ends, len(b) + 1)
    out, prev = [], 0
    for i in range(n):
        out.append(b[prev:ends[i]].decode("utf-8"))
        prev = ends[i]
    return out


@pytest.mark.parametrize("text", SAMPLES)
def test_pretokenizer_c_matches_python_port(lib, text):
    assert c_pretok(lib, text) == T.pretokenize(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_pretokenizer_matches_real_regex(text):
    pytest.importorskip("regex")
    assert T.pretokenize(text) == T.pretokenize_regex(text)


def test_pretokenizer_fuzz_vs_regex(lib):
    pytest.importorskip("regex")
    rng = random.Random(0)
    alphabet = list("abcXYZ019 '\t\n\r.,!?-_()") + ["é", "ß", "д", "你", "。", "😀", " ", "＋"]
    for _ in range(400):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        assert c_pretok(lib, s) == T.pretokenize(s) == T.pretokenize_regex(s), repr(s)


def test_pretokenizer_known_divergence_documented():
    """Scripts with combining marks are only approximated (see README)."""
    pytest.importorskip("regex")
    s = "नमस्ते"  # Devanagari: vowel signs are \p{M}, we classify them as letters
    assert T.pretokenize(s) != T.pretokenize_regex(s)


@pytest.fixture(scope="module")
def tok_model(lib, models):
    from conftest import Model
    m = Model(lib, models["f32"])
    g = read_gguf(models["f32"])
    ref = T.BPETokenizer(g.kv["tokenizer.ggml.tokens"], g.kv["tokenizer.ggml.merges"],
                         g.kv["tokenizer.ggml.token_type"])
    yield m, ref
    m.close()


@pytest.mark.parametrize("text", SAMPLES + ["<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"])
def test_encode_c_matches_python(tok_model, text):
    m, ref = tok_model
    assert m.tokenize(text) == ref.encode(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip(tok_model, text):
    m, _ = tok_model
    assert m.detokenize_bytes(m.tokenize(text)) == text.encode("utf-8")


def test_bpe_actually_merges(tok_model):
    m, ref = tok_model
    text = "Once upon a time there was a tiny inference engine."
    ids = m.tokenize(text)
    assert len(ids) < len(text.encode()) / 2  # trained merges compress the corpus text
    assert ref.decode(ids) == text


def test_special_tokens(tok_model):
    m, ref = tok_model
    im_end = ref.vocab["<|im_end|>"]
    ids = m.tokenize("a<|im_end|>b")
    assert im_end in ids
    assert m.detokenize_bytes(ids) == b"a<|im_end|>b"
    # with parse_special=False the marker is ordinary text
    plain = m.tokenize("a<|im_end|>b", parse_special=False)
    assert im_end not in plain
    assert m.detokenize_bytes(plain) == b"a<|im_end|>b"
    assert m.lib.ti_token_id(m.h, b"<|im_end|>") == im_end
    assert m.lib.ti_token_id(m.h, b"definitely-not-a-token") == -1


def test_fuzz_roundtrip_random_unicode(tok_model):
    m, ref = tok_model
    rng = random.Random(1)
    pool = [chr(c) for c in list(range(32, 127)) + list(range(0xA0, 0x250)) + list(range(0x400, 0x450))
            + list(range(0x4E00, 0x4E40)) + [0x1F600, 0x1F680, 0x3000, 0x2028, 10, 13, 9]]
    for _ in range(200):
        s = "".join(rng.choice(pool) for _ in range(rng.randint(1, 60)))
        ids = m.tokenize(s)
        assert ids == ref.encode(s)
        assert m.detokenize_bytes(ids) == s.encode("utf-8")


def test_utf8_split_across_tokens_streams_correctly(tok_model):
    """Byte-level tokens can split a multi-byte char; concatenated bytes must still be valid."""
    m, _ = tok_model
    s = "\U0001F680\U0001F680 Ж"
    ids = m.tokenize(s)
    joined = b"".join(m.detokenize_bytes([i]) for i in ids)
    assert joined.decode("utf-8") == s


def test_tokenize_buffer_too_small_reports_needed(lib, tok_model):
    m, _ = tok_model
    out = np.zeros(2, dtype=np.int32)
    from conftest import iptr
    n = lib.ti_tokenize(m.h, b"hello world, this is longer", iptr(out), 2, 1)
    assert n > 2
