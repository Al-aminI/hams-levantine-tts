"""English grapheme-to-phoneme, producing IPA from the shared inventory.

Primary backend is espeak-ng (high-quality, fast, ubiquitous).  A tiny rule fallback
keeps the pipeline importable and unit-testable on machines without espeak-ng, though
the server/training path assumes espeak-ng is present (it is installed by ``install.sh``).
"""

from __future__ import annotations

from typing import List

from . import espeak
from .phoneme_inventory import fold_to_inventory, tokenize_ipa

# espeak emits a handful of symbols we normalise toward the inventory before tokenising.
_REWRITE = [
    ("ɡ", "ɡ"),   # ensure IPA g (U+0261)
    ("g", "ɡ"),
    ("ɹ", "ɹ"),   # keep English rhotic distinct from Arabic r/ɾ
    ("ʔ", "ʔ"),
    ("ɐ", "ə"),
    ("ɚ", "ə"),
    ("ɝ", "ɜː"),
    ("ɫ", "ɫ"),
    ("oʊ", "o͡ʊ"), ("aʊ", "a͡ʊ"), ("aɪ", "a͡ɪ"), ("eɪ", "e͡ɪ"), ("ɔɪ", "ɔ͡ɪ"),
    ("dʒ", "d͡ʒ"), ("tʃ", "t͡ʃ"),
]


def _rewrite(ipa: str) -> str:
    for a, b in _REWRITE:
        ipa = ipa.replace(a, b)
    return ipa


def english_g2p(text: str) -> str:
    """Convert English text to an IPA string drawn from the shared inventory."""
    if espeak.available():
        ipa = espeak.phonemize(text, voice="en-us")
        ipa = _rewrite(ipa)
        # tokenise + re-render so anything OOV is folded, never dropped
        return "".join(
            s if s == " " else fold_to_inventory(s)
            for s in _retokenize(ipa)
        )
    return _fallback(text)


def _retokenize(ipa: str) -> List[str]:
    out: List[str] = []
    for word in ipa.split(" "):
        toks = tokenize_ipa(word).symbols
        if out:
            out.append(" ")
        out.extend(toks)
    return out


# ---- minimal letter-rule fallback (only used when espeak-ng is absent) ----
_FALLBACK_MAP = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f", "g": "ɡ",
    "h": "h", "i": "ɪ", "j": "d͡ʒ", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "ɒ", "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t", "u": "ʌ",
    "v": "v", "w": "w", "x": "ks", "y": "j", "z": "z",
}


def _fallback(text: str) -> str:
    words = []
    for word in text.lower().split():
        words.append("".join(_FALLBACK_MAP.get(c, "") for c in word))
    return " ".join(w for w in words if w)


if __name__ == "__main__":
    for t in ["hello world", "the real-time TTS engine", "Levantine code-switching"]:
        print(f"{t!r:32} -> {english_g2p(t)!r}")
