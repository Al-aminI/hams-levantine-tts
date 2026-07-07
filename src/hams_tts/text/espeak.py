"""Thin, dependency-free wrapper around the ``espeak-ng`` CLI for IPA phonemisation.

We shell out to ``espeak-ng --ipa=1`` rather than depend on the ``phonemizer`` Python
package: it removes a fragile dependency, works identically on the Mac dev box and the
L4 server, and ``--ipa=1`` gives us *unambiguous* boundaries on this build — phonemes
within a word are joined by ``_`` and words are separated by a single space.

espeak-ng is used for two roles:
  * the **English** G2P (its English is excellent), and
  * a **fallback Arabic** phonemiser when no Arabic diacritiser is installed (its MSA
    output is then surface-remapped toward Levantine in :mod:`levantine_g2p`).
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess

_ESPEAK = shutil.which("espeak-ng") or shutil.which("espeak")

# strip espeak's clause/markup artefacts and zero-width joiners it leaks into IPA
_PAREN = re.compile(r"\([^)]*\)")
_ZW = dict.fromkeys(map(ord, "‍‌​﻿"), None)


def available() -> bool:
    return _ESPEAK is not None


def binary() -> str | None:
    return _ESPEAK


@functools.lru_cache(maxsize=8192)
def phonemize(text: str, voice: str = "en-us") -> str:
    """Return a space-delimited IPA string (words separated by single spaces).

    Raises ``RuntimeError`` if espeak-ng is not installed so callers can fall back.
    """
    if not _ESPEAK:
        raise RuntimeError("espeak-ng not found on PATH")
    if not text.strip():
        return ""
    proc = subprocess.run(
        [_ESPEAK, "-v", voice, "--ipa=1", "-q", "--", text],
        capture_output=True,
        text=True,
        encoding="utf-8",   # espeak emits UTF-8 IPA; Windows default (cp1252) mangles it
        errors="replace",
        timeout=30,
    )
    raw = proc.stdout or ""
    raw = _PAREN.sub(" ", raw).translate(_ZW)
    # words: whitespace-separated; phonemes within a word: '_' separated
    words = []
    for w in raw.split():
        phon = w.replace("_", "").strip()
        if phon:
            words.append(phon)
    return " ".join(words)


if __name__ == "__main__":
    print("espeak available:", available(), binary())
    for t, v in [("hello world", "en-us"), ("real-time streaming", "en-us"), ("مَرْحَبا", "ar")]:
        print(f"{v:6} {t!r:28} -> {phonemize(t, v)!r}")
