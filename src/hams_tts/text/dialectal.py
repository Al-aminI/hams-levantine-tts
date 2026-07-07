"""Dialectal de-desinentialisation: strip MSA case/mood endings (iʿrab) that a MSA
diacritiser (camel-tools) restores but spoken Levantine does not realise.

Levantine Arabic is largely caseless — nouns/adjectives/verbs surface in a pausal-like
form. A MSA diacritiser gives e.g. مَوْعِدِ (moːʕid-**i**, genitive) where Levantine says
moːʕid. Feeding the raw MSA tashkeel to the Levantine G2P therefore injects final short
vowels the audio never has — a residual label↔audio mismatch. This module removes exactly
those word-final case markers, conservatively, leaving stem vowels and true dialectal
finals intact.

Rules (applied per whitespace token, only to the WORD-FINAL position):
  * bare final short vowel  فَتْحة/ضَمّة/كَسْرة  -> drop            (rafʕ/naṣb/jarr iʿrab)
  * final tanwin ٌ / ٍ  (nominative/genitive nunation)   -> drop
  * final tanwin ً after ة  (accusative noun: واقِفَةً)  -> drop the tanwin
  * final tanwin ً elsewhere (adverbial: شُكْرًا، طَبْعًا) -> KEEP  (pronounced -an)
  * long vowels ا/و/ي, shadda, sukun, and all non-final marks -> untouched

The transform is a no-op on undiacritised input and on words that already end pausally,
so it is safe to run unconditionally in front of the Levantine G2P.
"""

from __future__ import annotations

FATHA, KASRA, DAMMA = "َ", "ِ", "ُ"
SUKUN, SHADDA = "ْ", "ّ"
TANWIN_F, TANWIN_K, TANWIN_D = "ً", "ٍ", "ٌ"
TEH_MARBUTA = "ة"

_MARKS = {FATHA, KASRA, DAMMA, SUKUN, SHADDA, TANWIN_F, TANWIN_K, TANWIN_D}
_BARE_SHORT = {FATHA, KASRA, DAMMA}
_DROP_TANWIN = {TANWIN_K, TANWIN_D}  # nominative/genitive nunation always drops


def _strip_word(w: str) -> str:
    """Remove the final letter's case/mood mark, preserving everything else — crucially
    the ORIGINAL mark order (the Levantine G2P needs shadda immediately after its
    consonant to geminate, so we must NOT reorder to Unicode-canonical form)."""
    if not w:
        return w

    # peel trailing combining marks off the final letter (order-independent read)
    i = len(w) - 1
    while i >= 0 and w[i] in _MARKS:
        i -= 1
    if i < 0:
        return w
    letter, marks, prefix = w[i], set(w[i + 1:]), w[:i]

    has_shadda = SHADDA in marks
    has_sukun = SUKUN in marks
    keep_case = ""  # the case/nunation mark we keep on the final letter (if any)

    if marks & _DROP_TANWIN or (marks & _BARE_SHORT):
        pass  # drop it: case iʿrab / nominative-genitive nunation
    elif TANWIN_F in marks:
        # accusative noun ...ةً -> drop; adverbial -an elsewhere (شكرًا) -> keep
        keep_case = "" if letter == TEH_MARBUTA else TANWIN_F

    # shadda FIRST (right after the consonant) so the G2P still geminates, then any kept
    # case mark, then sukun.
    rebuilt = letter + (SHADDA if has_shadda else "") + keep_case + (SUKUN if has_sukun else "")
    return prefix + rebuilt


def strip_case_endings(text: str) -> str:
    """Remove MSA word-final case/mood markers so the Levantine G2P sees pausal forms."""
    return " ".join(_strip_word(w) for w in text.split())


if __name__ == "__main__":  # battery of known cases
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import unicodedata

    def norm(s):
        return unicodedata.normalize("NFC", s)
    cases = [
        ("مَوْعِدِ", "مَوْعِد", "genitive iʿrab dropped"),
        ("المَقالِ", "المَقال", "genitive dropped"),
        ("كِتابٌ", "كِتاب", "nominative nunation dropped"),
        ("واقِفَةً", "واقِفَة", "accusative ...ةً tanwin dropped"),
        ("شُكْرًا", "شُكْرًا", "adverbial -an KEPT"),
        ("طَبْعًا", "طَبْعًا", "adverbial -an KEPT"),
        ("بَيْتي", "بَيْتي", "long final -i kept"),
        ("هُوَ", "هُو", "pronoun final fatha dropped"),
        ("مُهِمٌّ", "مُهِمّ", "shadda kept, nunation dropped"),
        ("عَمِّ", "عَمّ", "shadda kept, case vowel dropped"),
    ]
    ok = 0
    for src, want, why in cases:
        got = strip_case_endings(src)
        mark = "OK " if got == norm(want) else "XX "
        ok += got == norm(want)
        print(f"{mark}{src!r:12} -> {got!r:12} (want {want!r:12}) {why}")
    print(f"\n{ok}/{len(cases)} passed")
