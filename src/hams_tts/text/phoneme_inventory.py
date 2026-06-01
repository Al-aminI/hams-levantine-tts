"""Unified IPA phoneme inventory for Levantine-Arabic / English code-switching TTS.

This module is the *interlingua* of the whole system.  Both the Arabic G2P and the
English G2P emit symbols drawn from a single, shared IPA inventory, and the acoustic
model is trained on token IDs derived from that inventory.  Because both languages
land in the *same* phonetic space, a code-switch boundary is just another phoneme
transition — there is no engine hand-off, and prosody can flow continuously across
the boundary.  Language identity is carried *separately* (see :data:`Lang`) and fed
to the model as an additive language-ID embedding, so the network can still colour
shared phonemes (e.g. /t/, /r/) with language-appropriate fine phonetics.

Design notes
------------
* IPA symbols are frequently multi-codepoint (tie bars ``͡``, length ``ː``,
  superscript pharyngealisation ``ˤ``).  Tokenisation therefore uses *longest-match*
  against the inventory rather than naive per-character splitting.
* The inventory is intentionally a *closed set*.  Anything a G2P emits that is not in
  the set is mapped to the nearest in-set symbol by :func:`fold_to_inventory` (with a
  logged warning in the front-end), guaranteeing the model never sees an OOV token.
* Stress (``ˈ`` ``ˌ``) and the syllable/word boundaries are kept as first-class
  symbols because they materially affect Levantine prosody and code-switch rhythm.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple


class Lang(IntEnum):
    """Language-ID stream values (fed to the model as an embedding index).

    PUNCT/NEUTRAL is its own id so that silence and boundary tokens are not
    spuriously attributed to either language by the conditioning network.
    """

    PAD = 0
    AR = 1  # Levantine Arabic
    EN = 2  # English
    NEUTRAL = 3  # punctuation, silence, boundaries, digits handled upstream


# --------------------------------------------------------------------------------------
# Special / structural symbols
# --------------------------------------------------------------------------------------
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
WORD_SEP = " "  # white space = word boundary (kept, it conditions prosody)
SYL_SEP = "."  # syllable boundary (optional, emitted by syllabifier)
# Punctuation that the acoustic model is allowed to "see" (drives pausing/prosody).
PUNCT = [",", ".", "?", "!", ";", ":", "…", "—", "(", ")", '"', "«", "»"]

SPECIALS = [PAD, BOS, EOS, UNK, WORD_SEP, SYL_SEP]

# --------------------------------------------------------------------------------------
# Suprasegmentals (order-independent, but must be in the inventory to be tokenisable)
# --------------------------------------------------------------------------------------
SUPRASEGMENTALS = [
    "ˈ",  # primary stress (U+02C8)
    "ˌ",  # secondary stress (U+02CC)
    "ː",  # length mark (U+02D0) -- only appears fused into long vowels below, but
          # kept standalone for robustness when an upstream tool emits it detached
]

# --------------------------------------------------------------------------------------
# Consonants -- union of the Levantine-Arabic and English inventories, in IPA.
# Multi-codepoint symbols (tie bars, pharyngealisation) are listed explicitly so the
# longest-match tokeniser can find them.
# --------------------------------------------------------------------------------------
CONSONANTS = [
    # --- shared / English-leaning ---
    "p", "b", "t", "d", "k", "ɡ",          # plosives (note: ɡ is U+0261, the IPA g)
    "t͡ʃ", "d͡ʒ",                            # English affricates
    "f", "v", "θ", "ð", "s", "z",          # fricatives
    "ʃ", "ʒ", "h", "x", "ɣ",               # more fricatives (x/ɣ shared w/ Arabic)
    "m", "n", "ŋ",                          # nasals
    "l", "ɫ",                               # laterals (ɫ = velarised/emphatic l, e.g. Allah)
    "r", "ɾ", "ɹ",                          # rhotics (Arabic trill/tap r/ɾ; English ɹ)
    "w", "j",                               # glides
    # --- Arabic-specific (Levantine) ---
    "ʔ",   # glottal stop  -> Levantine realisation of ق, and hamza
    "q",   # uvular plosive -> retained for MSA loans / Quranic register / some lexemes
    "ħ",   # voiceless pharyngeal fricative (ح)
    "ʕ",   # voiced pharyngeal fricative (ع)
    "sˤ",  # emphatic s (ص)
    "dˤ",  # emphatic d (ض)
    "tˤ",  # emphatic t (ط)
    "ðˤ",  # emphatic interdental / merges toward zˤ in Levant (ظ)
    "zˤ",  # emphatic z (Levantine realisation of ظ for many speakers)
]

# --------------------------------------------------------------------------------------
# Vowels -- Levantine has a richer surface inventory than MSA (phonemic e/o, schwa).
# Long vowels are stored as single fused symbols ("aː") for stable tokenisation.
# --------------------------------------------------------------------------------------
VOWELS = [
    # short
    "a", "i", "u", "e", "o", "ə",
    # English lax / extra qualities
    "ɪ", "ʊ", "ɛ", "æ", "ʌ", "ɑ", "ɒ", "ɔ", "ɜ", "ɐ",
    # long (Arabic + English tense)
    "aː", "iː", "uː", "eː", "oː", "ɑː", "ɔː", "ɜː",
    # diphthongs (English; Levantine aj/aw usually monophthongise -> handled in G2P)
    "a͡ɪ", "a͡ʊ", "e͡ɪ", "o͡ʊ", "ɔ͡ɪ",
]

# --------------------------------------------------------------------------------------
# Final ordered symbol table.  ORDER IS A FROZEN CONTRACT: the model's embedding rows
# are indexed by this order, so never reorder -- only append new symbols at the end.
# --------------------------------------------------------------------------------------
SYMBOLS: List[str] = (
    SPECIALS
    + PUNCT
    + SUPRASEGMENTALS
    + CONSONANTS
    + VOWELS
)

# De-duplicate while preserving order (defensive; the lists above are disjoint).
_seen: set = set()
_ordered: List[str] = []
for _s in SYMBOLS:
    if _s not in _seen:
        _seen.add(_s)
        _ordered.append(_s)
SYMBOLS = _ordered

SYMBOL_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL: Dict[int, str] = {i: s for s, i in SYMBOL_TO_ID.items()}
VOCAB_SIZE: int = len(SYMBOLS)

# Pre-sorted (by length desc, then lexicographically) view for longest-match tokenising.
_MULTI_FIRST: List[str] = sorted(
    [s for s in SYMBOLS if s not in (PAD, BOS, EOS, UNK)],
    key=lambda s: (-len(s), s),
)

# A small fold table: symbols a G2P might emit that we deliberately collapse to a
# canonical in-inventory symbol (keeps the model vocab tight without losing coverage).
FOLD_MAP: Dict[str, str] = {
    "g": "ɡ",          # ASCII g -> IPA ɡ
    "ɡ": "ɡ",
    "c": "k",
    "y": "j",
    "ʤ": "d͡ʒ",         # precomposed affricates -> tie-bar form
    "ʧ": "t͡ʃ",
    "dʒ": "d͡ʒ",        # untied -> tied (some espeak builds omit the tie bar)
    "tʃ": "t͡ʃ",
    "aɪ": "a͡ɪ",
    "aʊ": "a͡ʊ",
    "eɪ": "e͡ɪ",
    "oʊ": "o͡ʊ",
    "ɔɪ": "ɔ͡ɪ",
    "ɝ": "ɜː",         # rhotacised schwa (American) -> long mid-central
    "ɚ": "ə",
    "ɡ̃": "ŋ",
    "ʁ": "ɣ",          # uvular fricative variants -> ɣ
    "χ": "x",
    "ɹ̩": "ɹ",
    "ɫ̩": "ɫ",
    "ɐ": "ə",
}


@dataclass
class TokenizedPhonemes:
    """Result of tokenising an IPA string against the inventory."""

    symbols: List[str]
    ids: List[int]

    def __len__(self) -> int:  # convenience
        return len(self.ids)


def fold_to_inventory(symbol: str) -> str:
    """Map an arbitrary symbol to the nearest in-inventory symbol (or UNK)."""
    if symbol in SYMBOL_TO_ID:
        return symbol
    if symbol in FOLD_MAP:
        return FOLD_MAP[symbol]
    # Strip stray combining marks then retry once.
    stripped = symbol.replace("͡", "").replace("ː", "")
    if stripped in SYMBOL_TO_ID:
        return stripped
    if stripped in FOLD_MAP:
        return FOLD_MAP[stripped]
    return UNK


def tokenize_ipa(ipa: str) -> TokenizedPhonemes:
    """Greedy longest-match tokenisation of an IPA string into inventory symbols.

    Whitespace is normalised to the single WORD_SEP symbol.  Unknown runs are folded
    via :func:`fold_to_inventory` and, failing that, mapped to UNK (never dropped, so
    alignment with any external annotation stays intact).
    """
    out_symbols: List[str] = []
    i = 0
    n = len(ipa)
    while i < n:
        ch = ipa[i]
        if ch.isspace():
            if out_symbols and out_symbols[-1] != WORD_SEP:
                out_symbols.append(WORD_SEP)
            i += 1
            continue
        # try longest match starting at i
        matched = None
        for cand in _MULTI_FIRST:
            if cand and ipa.startswith(cand, i):
                matched = cand
                break
        if matched is not None:
            out_symbols.append(matched)
            i += len(matched)
        else:
            folded = fold_to_inventory(ch)
            out_symbols.append(folded)
            i += 1
    # collapse any accidental double word-seps and trim edges
    cleaned: List[str] = []
    for s in out_symbols:
        if s == WORD_SEP and (not cleaned or cleaned[-1] == WORD_SEP):
            continue
        cleaned.append(s)
    while cleaned and cleaned[-1] == WORD_SEP:
        cleaned.pop()
    ids = [SYMBOL_TO_ID[s] for s in cleaned]
    return TokenizedPhonemes(symbols=cleaned, ids=ids)


def encode(ipa: str, add_bos_eos: bool = True) -> Tuple[List[int], List[str]]:
    """Tokenise + (optionally) wrap with BOS/EOS.  Returns (ids, symbols)."""
    tok = tokenize_ipa(ipa)
    symbols = tok.symbols
    if add_bos_eos:
        symbols = [BOS] + symbols + [EOS]
    ids = [SYMBOL_TO_ID[s] for s in symbols]
    return ids, symbols


def decode(ids: List[int]) -> str:
    """Inverse of :func:`encode` for debugging / round-trip tests."""
    out = []
    for i in ids:
        s = ID_TO_SYMBOL.get(i, UNK)
        if s in (BOS, EOS, PAD):
            continue
        out.append(s)
    return "".join(out)


if __name__ == "__main__":  # quick self-check
    print(f"VOCAB_SIZE = {VOCAB_SIZE}")
    demo = "marħaba ساعدني e͡ɪ d͡ʒ"  # mixed-ish IPA-looking demo
    ids, syms = encode("ʔadeːʃ" )
    print("symbols:", syms)
    print("ids:", ids)
    print("round-trip:", decode(ids))
