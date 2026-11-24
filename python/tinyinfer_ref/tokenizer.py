"""Reference byte-level BPE (mirrors src/tokenizer.c) plus a tiny BPE trainer.

The pre-tokeniser is a line-by-line port of the C matcher so that tests can
compare C vs Python exactly; `QWEN2_PATTERN` is the real regex, used (via the
third-party `regex` module, if installed) to check the approximation.
"""
from collections import Counter

QWEN2_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                 r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE2U = bytes_to_unicode()
U2BYTE = {v: k for k, v in BYTE2U.items()}


# ---- unicode category approximation (identical to the C tables) ----------
def is_space(c):
    return (c == 32 or 0x09 <= c <= 0x0D or c in (0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F,
                                                    0x205F, 0x3000) or 0x2000 <= c <= 0x200A)


def is_number(c):
    if c < 0x80:
        return 48 <= c <= 57
    return (c in (0xB2, 0xB3, 0xB9, 0x2070, 0x3007) or 0xBC <= c <= 0xBE or 0x660 <= c <= 0x669
            or 0x6F0 <= c <= 0x6F9 or 0x966 <= c <= 0x96F or 0x2074 <= c <= 0x2079
            or 0x2080 <= c <= 0x2089 or 0x2150 <= c <= 0x2182 or 0x2185 <= c <= 0x2189
            or 0x2460 <= c <= 0x249B or 0x24EA <= c <= 0x24FF or 0x2776 <= c <= 0x2793
            or 0x3021 <= c <= 0x3029 or 0xFF10 <= c <= 0xFF19)


_NONLETTER = [(0x2C2, 0x2C5), (0x2D2, 0x2DF), (0x2E5, 0x2EB), (0x2EF, 0x36F), (0x482, 0x489),
              (0x591, 0x5CF), (0x600, 0x61F), (0x64B, 0x66D), (0x2000, 0x206F), (0x20A0, 0x20FF),
              (0x2190, 0x2BFF), (0x2E00, 0x2E7F), (0x2FF0, 0x3004), (0x3008, 0x3020),
              (0x302A, 0x3030), (0x3036, 0x303A), (0x303D, 0x303F), (0x3099, 0x309C),
              (0x3200, 0x33FF), (0x4DC0, 0x4DFF), (0xD800, 0xF8FF), (0xFE00, 0xFE6F),
              (0xFF00, 0xFF20), (0xFF3B, 0xFF40), (0xFF5B, 0xFF65), (0xFFE0, 0xFFFF),
              (0x1D000, 0x1D3FF), (0x1F000, 0x1FAFF), (0xE0000, 0x10FFFF)]
_LETTERLIKE = ({0x2102, 0x2107, 0x2115, 0x2124, 0x2126, 0x2128, 0x214E}
               | set(range(0x210A, 0x2114)) | set(range(0x2119, 0x211E))
               | set(range(0x212A, 0x212E)) | set(range(0x212F, 0x213A))
               | set(range(0x213C, 0x2140)) | set(range(0x2145, 0x214A)))


def is_letter(c):
    if c < 0x80:
        return 97 <= c <= 122 or 65 <= c <= 90
    if c < 0x100:
        return c in (0xAA, 0xB5, 0xBA) or (c >= 0xC0 and c not in (0xD7, 0xF7))
    if is_number(c) or is_space(c):
        return False
    if c in (0x375, 0x37E, 0x384, 0x385, 0x387, 0x30A0, 0x30FB):
        return False
    if 0x2100 <= c <= 0x214F:
        return c in _LETTERLIKE
    for lo, hi in _NONLETTER:
        if lo <= c <= hi:
            return False
    return True


def is_nl(c):
    return c in (10, 13)


def is_other(c):
    return not is_space(c) and not is_letter(c) and not is_number(c)


def pretokenize(text: str):
    cps = [ord(ch) for ch in text]
    n = len(cps)
    out = []
    i = 0
    while i < n:
        c = cps[i]
        j = -1
        if c == ord("'") and i + 1 < n:
            c1 = chr(cps[i + 1]).lower() if cps[i + 1] < 128 else ""
            c2 = chr(cps[i + 2]).lower() if i + 2 < n and cps[i + 2] < 128 else ""
            if c1 in ("s", "t", "m", "d"):
                j = i + 2
            elif (c1, c2) in (("r", "e"), ("v", "e"), ("l", "l")):
                j = i + 3
        if j < 0:
            k = -1
            if is_letter(c):
                k = i
            elif not is_nl(c) and not is_number(c) and i + 1 < n and is_letter(cps[i + 1]):
                k = i + 1
            if k >= 0:
                while k < n and is_letter(cps[k]):
                    k += 1
                j = k
        if j < 0 and is_number(c):
            j = i + 1
        if j < 0:
            k = i
            if c == 32 and i + 1 < n and is_other(cps[i + 1]):
                k = i + 1
            if k < n and is_other(cps[k]):
                while k < n and is_other(cps[k]):
                    k += 1
                while k < n and is_nl(cps[k]):
                    k += 1
                j = k
        if j < 0 and is_space(c):
            e = i
            while e < n and is_space(cps[e]):
                e += 1
            last_nl = max((k for k in range(i, e) if is_nl(cps[k])), default=-1)
            if last_nl >= 0:
                j = last_nl + 1
            elif e == n:
                j = e
            elif e - i >= 2:
                j = e - 1
            else:
                j = e
        if j < 0:
            j = i + 1
        out.append(text[i:j])
        i = j
    return out


def pretokenize_regex(text: str):
    import regex  # optional dependency, only for the approximation test
    return regex.findall(QWEN2_PATTERN, text)


def to_bytelevel(piece: str) -> str:
    return "".join(BYTE2U[b] for b in piece.encode("utf-8"))


# ---- BPE -------------------------------------------------------------------
class BPETokenizer:
    def __init__(self, tokens, merges, token_types=None):
        self.tokens = list(tokens)
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.types = list(token_types) if token_types else [1] * len(tokens)
        self.specials = sorted((i for i, t in enumerate(self.types) if t in (3, 4)),
                               key=lambda i: -len(self.tokens[i]))

    def _bpe(self, word: str):
        syms = list(word)
        while len(syms) > 1:
            best, best_rank = -1, None
            for i in range(len(syms) - 1):
                r = self.ranks.get(syms[i] + " " + syms[i + 1])
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best < 0:
                break
            syms[best:best + 2] = [syms[best] + syms[best + 1]]
        out = []
        for s in syms:
            if s in self.vocab:
                out.append(self.vocab[s])
            else:
                out.extend(self.vocab[ch] for ch in s if ch in self.vocab)
        return out

    def _encode_span(self, text):
        ids = []
        for p in pretokenize(text):
            ids.extend(self._bpe(to_bytelevel(p)))
        return ids

    def encode(self, text: str, parse_special=True):
        if not parse_special or not self.specials:
            return self._encode_span(text)
        ids, seg, i = [], 0, 0
        while i < len(text):
            hit = next((s for s in self.specials if text.startswith(self.tokens[s], i)), None)
            if hit is not None:
                ids += self._encode_span(text[seg:i]) + [hit]
                i += len(self.tokens[hit])
                seg = i
            else:
                i += 1
        return ids + self._encode_span(text[seg:])

    def decode_bytes(self, ids) -> bytes:
        out = bytearray()
        for i in ids:
            t = self.tokens[i]
            if self.types[i] in (3, 4):
                out += t.encode("utf-8")
            else:
                for ch in t:
                    out += bytes([U2BYTE[ch]]) if ch in U2BYTE else ch.encode("utf-8")
        return bytes(out)

    def decode(self, ids) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")


def train_bpe(corpus: str, n_merges: int, specials=SPECIALS):
    """Classic BPE training on byte-level pre-tokens. Returns (tokens, merges, types)."""
    words = Counter(tuple(to_bytelevel(p)) for p in pretokenize(corpus))
    base = [BYTE2U[b] for b in range(256)]
    tokens = list(base)
    merges = []
    words = {w: c for w, c in words.items()}
    for _ in range(n_merges):
        pairs = Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (a, b), cnt = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        if cnt < 2:
            break
        merges.append(a + " " + b)
        tokens.append(a + b)
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + c
        words = new
    types = [1] * len(tokens) + [3] * len(specials)
    return tokens + list(specials), merges, types
