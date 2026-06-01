"""Text normalisation for the Levantine/English code-switching front-end.

Responsibilities (everything *before* G2P):

* Unicode hygiene: NFC, strip tatweel/ZWJ/ZWNJ, fold presentation forms, canonicalise
  whitespace and the various Arabic punctuation marks to a small in-inventory set.
* Conservative Arabic letter normalisation.  Crucially we **do not** collapse the
  hamza-bearing alef forms (أ إ آ) to bare ا — for ASR that is standard, but for TTS it
  would destroy the glottal-stop / long-vowel distinction.  We only fix cosmetic
  variants (e.g. Arabic presentation forms, superscript alef rendering).
* Verbalisation of non-lexical tokens — numbers, ordinals, currency, percentages,
  times and years — into *spoken words* in the appropriate language.  Arabic numbers
  are emitted **already diacritised in Levantine colloquial form** for the high-frequency
  range (0-99) so they can flow straight into the Levantine G2P, with an MSA-style
  compositional fallback for larger magnitudes.

This module is deterministic and dependency-light (stdlib + optional ``num2words``),
so it runs and is unit-tested on CPU-only machines.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

try:  # optional, only used for English; we ship a fallback so it's never required
    from num2words import num2words as _num2words  # type: ignore

    _HAVE_NUM2WORDS = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_NUM2WORDS = False


# --------------------------------------------------------------------------------------
# Unicode constants
# --------------------------------------------------------------------------------------
TATWEEL = "ـ"          # ـ  Arabic kashida (purely decorative elongation)
ZWJ = "‍"              # zero-width joiner (espeak leaks these into IPA too)
ZWNJ = "‌"
ARABIC_DIATRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")

# Arabic-Indic and Extended Arabic-Indic digits -> ASCII
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})

# Punctuation canonicalisation: map many marks onto the small set the model knows.
_PUNCT_MAP = {
    "،": ",", "؍": ",", "٫": ".", "٬": ",",
    "؛": ";", "؟": "?", "٪": "%", "−": "-",
    "“": '"', "”": '"', "„": '"', "‟": '"', "’": "'", "‘": "'",
    "–": "—", "―": "—", "‒": "—",
    " ": " ", " ": " ", "\t": " ",
}
_PUNCT_TABLE = {ord(k): v for k, v in _PUNCT_MAP.items()}


# --------------------------------------------------------------------------------------
# Levantine colloquial cardinals (diacritised) for the high-frequency range.
# These are emitted ready for the Levantine G2P (no diacritiser needed for digits).
# --------------------------------------------------------------------------------------
_LEV_ONES = {
    0: "صِفِر", 1: "واحَد", 2: "تْنين", 3: "تْلاتة", 4: "أَربَعة",
    5: "خَمسة", 6: "سِتّة", 7: "سَبعة", 8: "تْمانية", 9: "تِسعة",
}
_LEV_TEENS = {
    10: "عَشَرة", 11: "إِحْداعَش", 12: "إِتْناعَش", 13: "تْلَتّاعَش",
    14: "أَربَعتاعَش", 15: "خَمستاعَش", 16: "سِتّاعَش", 17: "سَبعتاعَش",
    18: "تْمانتاعَش", 19: "تِسعتاعَش",
}
_LEV_TENS = {
    20: "عِشرين", 30: "تْلاتين", 40: "أَربعين", 50: "خَمسين",
    60: "سِتّين", 70: "سَبعين", 80: "تْمانين", 90: "تِسعين",
}
# MSA-ish scaffolding for hundreds/thousands (diacritised; broadly shared in Levant).
_AR_HUNDREDS = {
    1: "مية", 2: "ميتين", 3: "تْلاتمية", 4: "أَربَعمية", 5: "خَمسمية",
    6: "سِتّمية", 7: "سَبعمية", 8: "تْمانمية", 9: "تِسعمية",
}


def arabic_int_to_words(n: int) -> str:
    """Levantine-leaning verbalisation of a non-negative integer.

    Accurate Levantine colloquial forms for 0-99 (the speech-dominant range), with a
    compositional fallback (largely shared morphology) for hundreds/thousands/millions.
    Returns *diacritised* Arabic.
    """
    if n < 0:
        return "ناقِص " + arabic_int_to_words(-n)
    if n < 10:
        return _LEV_ONES[n]
    if n < 20:
        return _LEV_TEENS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _LEV_TENS[tens * 10]
        # Levantine: ones + "و" + tens, e.g. 21 = واحَد وعِشرين
        return f"{_LEV_ONES[ones]} وْ{_LEV_TENS[tens * 10]}"
    if n < 1000:
        hund, rem = divmod(n, 100)
        head = _AR_HUNDREDS[hund]
        if rem == 0:
            return head
        return f"{head} وْ{arabic_int_to_words(rem)}"
    if n < 1_000_000:
        thou, rem = divmod(n, 1000)
        if thou == 1:
            head = "أَلف"
        elif thou == 2:
            head = "أَلفين"
        elif thou <= 10:
            head = f"{arabic_int_to_words(thou)} تْلاف"
        else:
            head = f"{arabic_int_to_words(thou)} أَلف"
        if rem == 0:
            return head
        return f"{head} وْ{arabic_int_to_words(rem)}"
    # millions+
    mill, rem = divmod(n, 1_000_000)
    if mill == 1:
        head = "مِليون"
    elif mill == 2:
        head = "مِليونين"
    else:
        head = f"{arabic_int_to_words(mill)} مَلايين"
    if rem == 0:
        return head
    return f"{head} وْ{arabic_int_to_words(rem)}"


def english_int_to_words(n: int) -> str:
    """English verbalisation; uses num2words when present, else a stdlib fallback."""
    if _HAVE_NUM2WORDS:
        return _num2words(n)
    return _english_fallback(n)


_EN_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"), (100, "hundred")]


def _english_fallback(n: int) -> str:
    if n < 0:
        return "minus " + _english_fallback(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _EN_TENS[t] + (("-" + _EN_ONES[o]) if o else "")
    for value, name in _EN_SCALES:
        if n >= value:
            head, rem = divmod(n, value)
            out = f"{_english_fallback(head)} {name}"
            if rem:
                joiner = " and " if value == 100 or rem < 100 else " "
                out += joiner + _english_fallback(rem)
            return out
    return _EN_ONES[n]


# --------------------------------------------------------------------------------------
# Currency / symbol handling
# --------------------------------------------------------------------------------------
_CURRENCY = {
    "$": ("dollar", "dollars", "دولار"),
    "£": ("pound", "pounds", "جْنيه"),
    "€": ("euro", "euros", "يورو"),
    "₪": ("shekel", "shekels", "شيكِل"),
    "د.إ": ("dirham", "dirhams", "دِرهَم"),
    "ر.س": ("riyal", "riyals", "ريال"),
    "ل.ل": ("lira", "lira", "ليرة"),
    "ل.س": ("lira", "lira", "ليرة"),
    "ج.م": ("pound", "pounds", "جْنيه"),
}


def _verbalise_currency(symbol: str, amount: str, lang: str) -> str:
    sg, pl, ar = _CURRENCY.get(symbol, ("", "", ""))
    # split integer / fraction
    if "." in amount:
        whole, frac = amount.split(".", 1)
    else:
        whole, frac = amount, ""
    whole_i = int(whole) if whole.isdigit() else 0
    if lang == "ar":
        words = arabic_int_to_words(whole_i) + " " + ar
        if frac and int(frac):
            words += " وْ" + arabic_int_to_words(int(frac)) + " قِرش"
        return words
    unit = sg if whole_i == 1 else pl
    words = f"{english_int_to_words(whole_i)} {unit}"
    if frac and int(frac):
        words += f" and {english_int_to_words(int(frac))} cents"
    return words


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """NFC + cosmetic cleanup that is safe for TTS (preserves phonemic distinctions)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(TATWEEL, "").replace(ZWJ, "").replace(ZWNJ, "")
    text = text.translate(_PUNCT_TABLE)
    text = text.translate(_DIGIT_MAP)  # Arabic-Indic digits -> ASCII for the verbaliser
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel (used by language-ID / baseline espeak path)."""
    return ARABIC_DIATRITICS.sub("", text)


_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_CURRENCY_RE = re.compile(
    r"([$£€₪]|د\.إ|ر\.س|ل\.ل|ل\.س|ج\.م)\s*(\d+(?:[.,]\d+)?)"
    r"|(\d+(?:[.,]\d+)?)\s*([$£€₪])"
)
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def verbalize(text: str, lang: str) -> str:
    """Expand numbers / currency / percentages / times into spoken words for ``lang``.

    ``lang`` is the resolved language of the surrounding span ("ar" or "en"); this is
    decided by the code-switch segmenter so digits inherit the right phonology.
    """
    is_ar = lang == "ar"

    def _int_words(n: int) -> str:
        return arabic_int_to_words(n) if is_ar else english_int_to_words(n)

    # currency first (consumes its digits)
    def _cur(m: re.Match) -> str:
        if m.group(1):  # symbol before amount
            return " " + _verbalise_currency(m.group(1), m.group(2).replace(",", "."), lang) + " "
        return " " + _verbalise_currency(m.group(4), m.group(3).replace(",", "."), lang) + " "

    text = _CURRENCY_RE.sub(_cur, text)

    # percentages
    def _pct(m: re.Match) -> str:
        val = m.group(1)
        num = _float_words(val, is_ar)
        tail = "بالمية" if is_ar else "percent"
        return f" {num} {tail} "

    text = _PERCENT_RE.sub(_pct, text)

    # times HH:MM
    def _time(m: re.Match) -> str:
        h, mn = int(m.group(1)), int(m.group(2))
        if is_ar:
            base = "السّاعة " + _int_words(h)
            return base if mn == 0 else f"{base} وْ{_int_words(mn)} دْقيقة"
        base = f"{_int_words(h)} o'clock" if mn == 0 else f"{_int_words(h)} {mn:02d}"
        if mn != 0:
            base = f"{_int_words(h)} {_int_words(mn)}"
        return f" {base} "

    text = _TIME_RE.sub(_time, text)

    # bare numbers (integers and decimals)
    def _num(m: re.Match) -> str:
        return " " + _float_words(m.group(0).replace(",", "."), is_ar) + " "

    text = _NUMBER_RE.sub(_num, text)

    return re.sub(r"\s+", " ", text).strip()


def _float_words(token: str, is_ar: bool) -> str:
    neg = token.startswith("-")
    token = token.lstrip("-")
    if "." in token:
        whole, frac = token.split(".", 1)
    else:
        whole, frac = token, ""
    whole_i = int(whole) if whole else 0
    words = arabic_int_to_words(whole_i) if is_ar else english_int_to_words(whole_i)
    if frac:
        point = "فاصِل" if is_ar else "point"
        digit_words = " ".join(
            (arabic_int_to_words(int(d)) if is_ar else english_int_to_words(int(d)))
            for d in frac
        )
        words = f"{words} {point} {digit_words}"
    if neg:
        words = ("ناقِص " if is_ar else "minus ") + words
    return words


def normalize(text: str, lang: Optional[str] = None) -> str:
    """Convenience: unicode-normalise then (if ``lang`` given) verbalise."""
    text = normalize_unicode(text)
    if lang is not None:
        text = verbalize(text, lang)
    return text


if __name__ == "__main__":
    for s, l in [
        ("عندي ٣ كتب و$5 بس", "ar"),
        ("I have 3 books and it costs $5.50", "en"),
        ("الساعة 3:30 المعدل 95%", "ar"),
        ("the meeting is at 3:30 and we hit 95%", "en"),
    ]:
        print(f"[{l}] {s!r}\n   -> {normalize(s, l)!r}")
