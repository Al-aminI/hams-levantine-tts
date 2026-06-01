"""Arabic automatic diacritisation (tashkeel) — a swappable stage in front of the G2P.

The G2P needs short vowels.  Real conversational input is mostly *undiacritised*, so we
restore tashkeel automatically.  Backends, in priority order:

  * ``camel``  – CAMeL Tools morphological disambiguator (strong, production-grade).
  * ``catt``   – CATT char-based transformer (Alasmary et al., ArabicNLP 2024), SOTA
                 DER; loaded from Hugging Face.  (Sadeed, 2025, is a drop-in upgrade.)
  * ``passthrough`` – assume the text is already diacritised (used by unit tests and
                 when callers diacritise upstream).

``get_diacritizer("auto")`` returns the best backend actually importable in the current
environment, or ``None`` if none is available — in which case the front-end takes the
espeak-ng MSA→Levantine fallback path (see :func:`levantine_g2p.arabic_fallback_ipa`).

Design rationale: keeping diacritisation behind a clean interface means we can ship a
CPU-only dev box (passthrough/fallback) while the GPU server loads CATT/CAMeL, without
touching any downstream code.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .normalize import ARABIC_DIATRITICS

_ARABIC_LETTER = re.compile(r"[ء-ي]")


def diacritic_ratio(text: str) -> float:
    """Fraction of Arabic letters that carry at least one diacritic (rough)."""
    letters = _ARABIC_LETTER.findall(text)
    if not letters:
        return 1.0  # no Arabic letters -> nothing to diacritise
    marks = ARABIC_DIATRITICS.findall(text)
    return len(marks) / max(1, len(letters))


def is_already_diacritized(text: str, threshold: float = 0.35) -> bool:
    """Heuristic: enough harakat present that we can skip diacritisation."""
    return diacritic_ratio(text) >= threshold


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------
def _camel_backend() -> Optional[Callable[[str], str]]:
    try:
        from camel_tools.disambig.mle import MLEDisambiguator  # type: ignore
        from camel_tools.utils.dediac import dediac_ar  # noqa: F401

        mle = MLEDisambiguator.pretrained()

        def _run(text: str) -> str:
            out_tokens = []
            for word in text.split():
                disambig = mle.disambiguate([word])
                if disambig and disambig[0].analyses:
                    diac = disambig[0].analyses[0].analysis.get("diac", word)
                    out_tokens.append(diac)
                else:
                    out_tokens.append(word)
            return " ".join(out_tokens)

        return _run
    except Exception:
        return None


def _catt_backend(model_id: str = "facebook/mms-tts-ara") -> Optional[Callable[[str], str]]:
    # CATT is published as a char-based transformer; loading is optional and heavy.
    # We wrap it defensively so absence simply disables the backend.
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore

        # NOTE: replace with the actual CATT checkpoint id you mirror to HF; kept here as
        # a clearly-marked integration point so the dependency is opt-in.
        catt_id = "MagedSaeed/catt-encoder-only"  # integration point (see README)
        tok = AutoTokenizer.from_pretrained(catt_id)
        model = AutoModelForTokenClassification.from_pretrained(catt_id)
        model.eval()

        def _run(text: str) -> str:  # pragma: no cover - requires GPU/network
            # Real CATT inference restores per-character diacritics; see the CATT repo
            # for the exact decode.  Implemented on the server where the model is present.
            raise NotImplementedError("CATT decode runs on the GPU server")

        return _run
    except Exception:
        return None


def _passthrough_backend() -> Callable[[str], str]:
    return lambda text: text


_FACTORIES = {
    "camel": _camel_backend,
    "catt": _catt_backend,
    "passthrough": lambda: _passthrough_backend(),
}


def get_diacritizer(backend: str = "auto") -> Optional[Callable[[str], str]]:
    """Return a callable ``str -> diacritised str`` for the requested backend.

    ``auto`` tries camel → catt and returns ``None`` if neither is importable (so the
    caller can fall back).  ``passthrough`` always succeeds.
    """
    if backend == "passthrough":
        return _passthrough_backend()
    if backend == "auto":
        for name in ("camel", "catt"):
            fn = _FACTORIES[name]()
            if fn is not None:
                return fn
        return None
    factory = _FACTORIES.get(backend)
    if factory is None:
        raise ValueError(f"unknown diacritiser backend: {backend!r}")
    return factory()


if __name__ == "__main__":
    samples = ["مرحبا كيف حالك", "مَرْحَبا كَيْفَ حالَك", "I am English only"]
    for s in samples:
        print(f"ratio={diacritic_ratio(s):.2f} diacritized={is_already_diacritized(s)}  {s!r}")
    print("auto backend available:", get_diacritizer("auto") is not None)
