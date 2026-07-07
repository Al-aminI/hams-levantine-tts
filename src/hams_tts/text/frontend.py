"""The unified text front-end: ``str -> (phoneme_ids, language_ids)``.

This is the public entry point of Section 1.  It chains:

    normalise → segment (code-switch) → per-span verbalise → per-span G2P
    → tokenise into the shared IPA inventory → emit a phoneme-ID stream and a
      *parallel* language-ID stream.

Why two parallel streams?  The acoustic model consumes phoneme IDs (the shared IPA
space gives smooth cross-language transitions) **plus** a per-token language ID that is
turned into an additive embedding.  This lets one model colour shared phonemes (e.g.
/t/, /r/, /l/) with language-appropriate fine phonetics and switch instantly at a
boundary — with no engine hand-off and no inserted silence, which is exactly the
"smooth phoneme transitions, no abrupt breaks" the brief asks for.

The class is cheap to construct and threadsafe-by-immutability after init; the server
holds a single shared instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .codeswitch import Span, segment
from .diacritize import get_diacritizer, is_already_diacritized
from .english_g2p import english_g2p
from .levantine_g2p import arabic_fallback_ipa, levantine_g2p
from .normalize import normalize_unicode, verbalize
from .phoneme_inventory import (
    BOS,
    EOS,
    PUNCT,
    SYL_SEP,
    SYMBOL_TO_ID,
    WORD_SEP,
    Lang,
    tokenize_ipa,
)

_PUNCT_SET = set(PUNCT)


@dataclass
class Utterance:
    """Everything downstream stages need, plus debug breadcrumbs."""

    text: str
    normalized: str
    ipa: str
    symbols: List[str]
    phoneme_ids: List[int]
    language_ids: List[int]
    spans: List[Span] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.phoneme_ids)

    def pretty(self) -> str:
        rows = [f"text       : {self.text}", f"normalized : {self.normalized}", f"ipa        : {self.ipa}"]
        rows.append("tokens     : " + " ".join(self.symbols))
        rows.append("lang_ids   : " + " ".join(Lang(l).name[0] if l in (1, 2) else "·" for l in self.language_ids))
        return "\n".join(rows)


class TextFrontend:
    def __init__(
        self,
        diacritizer_backend: str = "auto",
        default_lang: str = "en",
        back_emphatics: bool = True,
        epenthesize: bool = True,
        dialectal_case_drop: bool = True,
    ) -> None:
        self.default_lang = default_lang
        self.back_emphatics = back_emphatics
        self.epenthesize = epenthesize
        # Strip MSA case/mood endings (iʿrab) a MSA diacritiser restores but Levantine
        # doesn't pronounce -- removes a residual label↔audio mismatch. See dialectal.py.
        self.dialectal_case_drop = dialectal_case_drop
        # may be None on a CPU dev box -> Arabic uses the espeak fallback path
        self._diacritizer = get_diacritizer(diacritizer_backend)
        self.diacritizer_backend = diacritizer_backend
        self.has_diacritizer = self._diacritizer is not None

    # -- Arabic span -> IPA, choosing the best available path --
    def _arabic_to_ipa(self, text: str) -> str:
        if is_already_diacritized(text):
            diac = text
        elif self._diacritizer is not None:
            diac = self._diacritizer(text)
        else:
            # no diacritiser available and text is bare -> espeak MSA -> Levantine remap
            return arabic_fallback_ipa(text, self.back_emphatics)
        if self.dialectal_case_drop:
            from .dialectal import strip_case_endings

            diac = strip_case_endings(diac)
        return levantine_g2p(diac, self.back_emphatics, self.epenthesize)

    def _span_to_ipa(self, span: Span) -> str:
        verbalized = verbalize(span.text, span.lang)
        if span.lang == "ar":
            return self._arabic_to_ipa(verbalized)
        return english_g2p(verbalized)

    def process(self, text: str) -> Utterance:
        normalized = normalize_unicode(text)
        spans = segment(normalized, default_lang=self.default_lang)

        symbols: List[str] = [BOS]
        language_ids: List[int] = [int(Lang.PAD)]
        ipa_parts: List[str] = []

        for idx, span in enumerate(spans):
            if idx > 0:  # word boundary between spans (neutral)
                symbols.append(WORD_SEP)
                language_ids.append(int(Lang.NEUTRAL))
            ipa = self._span_to_ipa(span)
            ipa_parts.append(ipa)
            span_lang = Lang.AR if span.lang == "ar" else Lang.EN
            for s in tokenize_ipa(ipa).symbols:
                symbols.append(s)
                if s in _PUNCT_SET or s in (WORD_SEP, SYL_SEP):
                    language_ids.append(int(Lang.NEUTRAL))
                else:
                    language_ids.append(int(span_lang))

        symbols.append(EOS)
        language_ids.append(int(Lang.PAD))

        phoneme_ids = [SYMBOL_TO_ID[s] for s in symbols]
        ipa = "  ".join(ipa_parts)
        assert len(phoneme_ids) == len(language_ids) == len(symbols)
        return Utterance(
            text=text,
            normalized=normalized,
            ipa=ipa,
            symbols=symbols,
            phoneme_ids=phoneme_ids,
            language_ids=language_ids,
            spans=spans,
        )

    __call__ = process


# Module-level convenience singleton for quick use / tests.
_DEFAULT: Optional[TextFrontend] = None


def get_default_frontend() -> TextFrontend:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TextFrontend()
    return _DEFAULT


if __name__ == "__main__":
    fe = TextFrontend()
    print(f"diacritizer backend present: {fe.has_diacritizer}\n")
    for t in [
        "مَرحَبا كيف حالَك؟",
        "بَدّي إحجِز flight من بيروت to London بُكرا الساعة 9",
        "Hams AI بَتِشتِغِل real-time عَ النيرو L4",
    ]:
        u = fe.process(t)
        print(u.pretty())
        print(f"n_tokens={len(u)}\n")
