"""Language segmentation for code-switched text.

Splits an arbitrary Arabic/English/mixed string into a list of ``(text, lang)`` spans
where ``lang`` is ``"ar"`` or ``"en"``.  Script is the primary signal (Arabic block vs
Latin); *neutral* runs (numbers, punctuation, whitespace) are attached to the language
of their nearest lexical neighbour so that, e.g., ``"عندي 3 books"`` segments as
``[("عندي 3", ar), ("books", en)]`` and the digit ``3`` is verbalised as Levantine
``تلاتة`` while ``books`` stays English.

Per-character script detection (rather than a statistical LID model) is deliberate: it
is deterministic, dependency-free, sub-millisecond, and exactly right for script-based
code-switching, which is what matters for a streaming TTS front-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


def _is_arabic(c: str) -> bool:
    o = ord(c)
    return (
        0x0600 <= o <= 0x06FF  # Arabic
        or 0x0750 <= o <= 0x077F  # Arabic Supplement
        or 0x08A0 <= o <= 0x08FF  # Arabic Extended-A
        or 0xFB50 <= o <= 0xFDFF  # Presentation Forms-A
        or 0xFE70 <= o <= 0xFEFF  # Presentation Forms-B
    )


def _is_latin(c: str) -> bool:
    o = ord(c)
    return (
        0x0041 <= o <= 0x005A
        or 0x0061 <= o <= 0x007A
        or 0x00C0 <= o <= 0x024F  # Latin-1 Supplement + Extended-A/B
    )


def _char_lang(c: str) -> Optional[str]:
    if _is_arabic(c):
        return "ar"
    if _is_latin(c):
        return "en"
    return None  # neutral: digits, punctuation, whitespace, symbols


@dataclass
class Span:
    text: str
    lang: str  # "ar" | "en"


def segment(text: str, default_lang: str = "en") -> List[Span]:
    """Return merged, language-tagged spans covering ``text`` in order."""
    if not text:
        return []

    # 1) group consecutive chars of identical (coarse) language signal
    raw: List[Tuple[str, Optional[str]]] = []
    cur_chars: List[str] = []
    cur_lang: Optional[str] = _char_lang(text[0])
    for c in text:
        cl = _char_lang(c)
        if cl == cur_lang:
            cur_chars.append(c)
        else:
            raw.append(("".join(cur_chars), cur_lang))
            cur_chars = [c]
            cur_lang = cl
    raw.append(("".join(cur_chars), cur_lang))

    # 2) resolve neutral runs to nearest lexical neighbour (prev preferred, then next)
    resolved: List[Optional[str]] = [lang for _, lang in raw]
    last_seen: Optional[str] = None
    for i, lang in enumerate(resolved):
        if lang is not None:
            last_seen = lang
        else:
            resolved[i] = last_seen
    next_seen: Optional[str] = None
    for i in range(len(resolved) - 1, -1, -1):
        if raw[i][1] is not None:
            next_seen = raw[i][1]
        elif resolved[i] is None:
            resolved[i] = next_seen
    resolved = [r if r is not None else default_lang for r in resolved]

    # 3) merge consecutive spans sharing the resolved language
    spans: List[Span] = []
    for (txt, _), lang in zip(raw, resolved):
        if spans and spans[-1].lang == lang:
            spans[-1] = Span(spans[-1].text + txt, lang)
        else:
            spans.append(Span(txt, lang))

    # 4) tidy whitespace; drop spans that are empty after trimming
    out: List[Span] = []
    for s in spans:
        t = s.text.strip()
        if t:
            out.append(Span(t, s.lang))
    return out


if __name__ == "__main__":
    examples = [
        "مرحبا، كيف حالك؟",
        "عندي meeting الساعة 3:30 بعد الظهر",
        "the app اسمها Hams ai بتشتغل real-time",
        "بدي أحجز flight من بيروت to London بكرا",
    ]
    for e in examples:
        print(f"\n{e}")
        for sp in segment(e):
            print(f"   [{sp.lang}] {sp.text!r}")
