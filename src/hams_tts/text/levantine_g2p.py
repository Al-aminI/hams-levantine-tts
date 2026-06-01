"""Rule-based grapheme-to-phoneme (G2P) for **Levantine Arabic**, emitting IPA from the
shared inventory in :mod:`hams_tts.text.phoneme_inventory`.

Why a *custom* G2P rather than espeak-ng's Arabic voice?  espeak-ng (and most MSA G2P)
(a) drops short vowels on under-diacritised input (`مرحبا` → `mrħbaː`), and (b) renders
*Modern Standard Arabic* phonology, not Levantine.  The two most audible Levantine
signatures — ق as a glottal stop /ʔ/ and ج as /ʒ/ — are simply wrong in an MSA G2P.
Owning the G2P lets us encode the dialect explicitly *and* keep it testable on CPU.

Pipeline (per whitespace token):
  1. small dialect **lexicon** override for high-frequency function words;
  2. **definite-article** handling with sun-/moon-letter assimilation (gemination);
  3. left-to-right walk with look-ahead for long vowels and the Levantine
     **diphthong monophthongisation** (aw→oː, ay→eː);
  4. post-processing: **emphatic backing** (a→ɑ around ص/ض/ط/ظ/ق-emphasis),
     teh-marbuta quality (ة→/e/, →/a/ after gutturals/emphatics), and a light
     CCC **epenthesis** heuristic.

The G2P consumes *diacritised* Arabic.  Diacritisation is a separate, swappable stage
(:mod:`hams_tts.text.diacritize`) so each concern is independently testable.

Known limitations are documented in the design doc; this is a strong, transparent
rule set, not a claim of perfect dialectal coverage.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ----------------------------------------------------------------------------------
# Arabic Unicode pieces
# ----------------------------------------------------------------------------------
FATHA, KASRA, DAMMA = "َ", "ِ", "ُ"
SUKUN, SHADDA = "ْ", "ّ"
TANWIN_F, TANWIN_K, TANWIN_D = "ً", "ٍ", "ٌ"
DAGGER_ALEF = "ٰ"
HARAKAT = {FATHA, KASRA, DAMMA, SUKUN, SHADDA, TANWIN_F, TANWIN_K, TANWIN_D, DAGGER_ALEF}

ALEF, ALEF_MAQSURA, ALEF_MADDA = "ا", "ى", "آ"
ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW = "أ", "إ"
HAMZA, WAW_HAMZA, YEH_HAMZA = "ء", "ؤ", "ئ"
TEH_MARBUTA, HAMZAT_WASL = "ة", "ٱ"
WAW, YEH, LAM = "و", "ي", "ل"

# ----------------------------------------------------------------------------------
# Consonant map -> Levantine IPA.  The two headline dialect rules live here:
#   ق -> ʔ   (urban Levantine glottal stop)
#   ج -> ʒ   (Levantine post-alveolar fricative, not the MSA affricate /d͡ʒ/)
# Interdentals ث/ذ/ظ follow urban Levantine stop/emphatic mergers.
# ----------------------------------------------------------------------------------
CONS = {
    "ب": "b",        # ب
    "ت": "t",        # ت
    "ث": "t",        # ث  -> /t/ in urban Levantine (stop merger; /s/ in some loans)
    "ج": "ʒ",        # ج  -> /ʒ/  (Levantine)
    "ح": "ħ",        # ح
    "خ": "x",        # خ
    "د": "d",        # د
    "ذ": "d",        # ذ  -> /d/ in urban Levantine (stop merger; /z/ in some loans)
    "ر": "r",        # ر
    "ز": "z",        # ز
    "س": "s",        # س
    "ش": "ʃ",        # ش
    "ص": "sˤ",       # ص
    "ض": "dˤ",       # ض
    "ط": "tˤ",       # ط
    "ظ": "zˤ",       # ظ  -> emphatic /zˤ/ (urban Levantine; MSA /ðˤ/)
    "ع": "ʕ",        # ع
    "غ": "ɣ",        # غ
    "ف": "f",        # ف
    "ق": "ʔ",        # ق  -> /ʔ/  (urban Levantine)
    "ك": "k",        # ك
    "ل": "l",        # ل
    "م": "m",        # م
    "ن": "n",        # ن
    "ه": "h",        # ه
    "ة": "t",        # ة  (only when NOT word-final; final handled specially)
    HAMZA: "ʔ", WAW_HAMZA: "ʔ", YEH_HAMZA: "ʔ",
}

SUN_LETTERS = set("تثدذرزسشصضطظلن")
EMPHATICS_IPA = {"sˤ", "dˤ", "tˤ", "zˤ"}
GUTTURALS_IPA = {"ħ", "ʕ", "x", "ɣ", "ʔ", "q", "h"}
CONSONANT_IPA = {
    "b", "t", "d", "k", "ɡ", "q", "ʔ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "x", "ɣ", "ħ", "ʕ", "m", "n", "ŋ", "l", "ɫ", "r", "ɾ", "ɹ", "w", "j",
    "sˤ", "dˤ", "tˤ", "zˤ", "t͡ʃ", "d͡ʒ",
}

# ----------------------------------------------------------------------------------
# High-frequency Levantine function-word lexicon (matched on the *undiacritised* form).
# These are the words a rule engine most often gets wrong; pinning them buys a lot of
# perceived naturalness for conversational agents.  Extend freely.
# ----------------------------------------------------------------------------------
LEXICON = {
    "الله": "ʔaɫɫa",
    "هذا": "haːda", "هاد": "haːd", "هيدا": "heːda",
    "هذه": "haːde", "هاي": "haj", "هيدي": "heːde",
    "ذلك": "haˈdaːk", "هداك": "haˈdaːk",
    "الذي": "ʔilli", "اللي": "ʔilli",
    "شو": "ʃuː", "ليش": "leːʃ", "كيف": "kiːf", "وين": "weːn",
    "هلق": "hallaʔ", "هلأ": "hallaʔ", "هسا": "hassa",
    "مش": "miʃ", "مو": "muː",
    "بدي": "ˈbiddi", "بدك": "ˈbiddak", "بدنا": "ˈbidna",
    "كتير": "ktiːr", "هيك": "heːk", "هون": "hoːn",
    "إيه": "ʔeː", "أيوا": "ʔaˈjwa", "لأ": "laʔ",
    "عشان": "ʕaˈʃaːn", "منشان": "minˈʃaːn", "بس": "bas",
    "في": "fiː", "مين": "miːn", "إمتى": "ʔemta", "قديش": "ʔaˈdeːʃ", "أديش": "ʔaˈdeːʃ",
}

_AR_LETTER_RE = re.compile(r"[ء-يٱ]")


def _strip_haraka(s: str) -> str:
    return "".join(c for c in s if c not in HARAKAT and c != "ـ")


def _apply_article(word: str) -> Tuple[List[str], str, bool]:
    """Detect a leading definite article and return (prefix_phones, remainder, geminate).

    Sun letters trigger lam-assimilation (the lam drops, the next consonant geminates);
    moon letters keep /l/.  We render the article vowel as /i/ (urban Levantine /il-/).
    """
    bare = _strip_haraka(word)
    if bare.startswith("ال") or word.startswith(HAMZAT_WASL + LAM) or bare.startswith("ٱل"):
        # remove the alef/wasl seat + lam from the *diacritised* string, robustly
        # find the lam, then the first real letter after it
        # locate index of LAM in original word
        li = word.find(LAM)
        if li == -1:
            return [], word, False
        rest = word[li + 1 :]
        # skip any diacritics directly after lam
        j = 0
        while j < len(rest) and rest[j] in HARAKAT:
            j += 1
        rest = rest[j:]
        m = _AR_LETTER_RE.search(rest)
        if not m:
            return ["ʔ", "i", "l"], rest, False
        first = rest[m.start()]
        if first in SUN_LETTERS:
            return ["ʔ", "i"], rest, True  # geminate handled by caller
        return ["ʔ", "i", "l"], rest, False
    return [], word, False


def _g2p_word(word: str) -> List[str]:
    bare = _strip_haraka(word)
    if bare in LEXICON:
        # lexical entries are already IPA strings; tokenise them loosely on known marks
        from .phoneme_inventory import tokenize_ipa

        return tokenize_ipa(LEXICON[bare]).symbols

    prefix, word, geminate_first = _apply_article(word)
    phones: List[str] = list(prefix)
    chars = list(word)
    n = len(chars)
    i = 0
    first_cons_emitted = not geminate_first  # if no article gemination needed, skip flag

    def _emit_cons(c_ipa: str) -> None:
        nonlocal first_cons_emitted
        phones.append(c_ipa)
        if not first_cons_emitted:
            phones.append(c_ipa)  # gemination from sun-letter assimilation
            first_cons_emitted = True

    while i < n:
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < n else ""
        nxt2 = chars[i + 2] if i + 2 < n else ""

        # ---- alef / madda / hamza-carriers (vowel & glottal seats) ----
        if ch == ALEF_MADDA:  # آ = hamza + long aː
            phones += ["ʔ", "aː"]
            i += 1
            continue
        if ch in (ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW):
            phones.append("ʔ")
            # the carried haraka follows as a normal vowel
            v = _vowel_for(nxt)
            if v:
                phones.append(v)
                i += 2
                continue
            phones.append("i" if ch == ALEF_HAMZA_BELOW else "a")
            i += 1
            continue
        if ch == ALEF or ch == HAMZAT_WASL:
            if i == 0:  # word-initial seat: elide (hamzat wasl) or light onset
                v = _vowel_for(nxt)
                if v:
                    phones += ["ʔ", v]
                    i += 2
                    continue
                i += 1
                continue
            phones.append("aː")  # medial bare alef = long /aː/
            i += 1
            continue
        if ch == ALEF_MAQSURA:
            phones.append("aː")
            i += 1
            continue

        # ---- teh marbuta ----
        if ch == TEH_MARBUTA:
            if i == n - 1 or (i == n - 2 and chars[i + 1] in HARAKAT):
                # final: /a/ after guttural/emphatic context else /e/ (Levantine imala)
                prev = _last_phone(phones)
                phones.append("a" if (prev in EMPHATICS_IPA or prev in {"ʕ", "ħ", "x", "ɣ", "q", "r"}) else "e")
            else:
                phones.append("t")
            i += 1
            continue

        # ---- waw / yeh as consonants (long-vowel uses are consumed by look-ahead) ----
        if ch == WAW:
            phones.append("w")
            i += 1
            i = _consume_haraka(chars, i, phones)
            continue
        if ch == YEH:
            phones.append("j")
            i += 1
            i = _consume_haraka(chars, i, phones)
            continue

        # ---- ordinary consonants ----
        if ch in CONS:
            c_ipa = CONS[ch]
            _emit_cons(c_ipa)
            i += 1
            # shadda -> gemination
            if i < n and chars[i] == SHADDA:
                phones.append(c_ipa)
                i += 1
            i = _consume_haraka(chars, i, phones)
            continue

        # ---- standalone diacritics or unknowns: skip ----
        i += 1

    return phones


def _vowel_for(mark: str) -> str:
    return {FATHA: "a", KASRA: "i", DAMMA: "u"}.get(mark, "")


def _last_phone(phones: List[str]) -> str:
    return phones[-1] if phones else ""


def _consume_haraka(chars: List[str], i: int, phones: List[str]) -> int:
    """Consume the haraka following a just-emitted consonant, applying Levantine
    long-vowel and diphthong-monophthongisation rules.  Returns the new index."""
    n = len(chars)
    if i >= n:
        return i
    mark = chars[i]
    nxt = chars[i + 1] if i + 1 < n else ""

    if mark == FATHA:
        # diphthong monophthongisation: aw -> oː, ay -> eː
        if nxt == WAW and _is_glide(chars, i + 1):
            phones.append("oː")
            return i + 2
        if nxt == YEH and _is_glide(chars, i + 1):
            phones.append("eː")
            return i + 2
        if nxt in (ALEF, ALEF_MAQSURA, ALEF_MADDA):
            phones.append("aː")
            return i + 2
        phones.append("a")
        return i + 1
    if mark == KASRA:
        if nxt == YEH and _is_glide(chars, i + 1):
            phones.append("iː")
            return i + 2
        phones.append("i")
        return i + 1
    if mark == DAMMA:
        if nxt == WAW and _is_glide(chars, i + 1):
            phones.append("uː")
            return i + 2
        phones.append("u")
        return i + 1
    if mark == SUKUN:
        return i + 1
    if mark == DAGGER_ALEF:
        phones.append("aː")
        return i + 1
    if mark in (TANWIN_F, TANWIN_K, TANWIN_D):
        # Levantine drops case nunation; keep only the vowel colour
        phones.append({TANWIN_F: "a", TANWIN_K: "i", TANWIN_D: "u"}[mark])
        return i + 1
    return i  # no haraka to consume


def _is_glide(chars: List[str], idx: int) -> bool:
    """A waw/yeh is a glide (part of a long vowel/diphthong) when it does not itself
    carry a full vowel haraka."""
    nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
    return nxt in ("", SUKUN) or nxt not in (FATHA, KASRA, DAMMA)


# ----------------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------------
def _emphatic_backing(phones: List[str]) -> List[str]:
    """Back /a, aː/ to /ɑ, ɑː/ in the neighbourhood of an emphatic consonant."""
    out = list(phones)
    for idx, p in enumerate(out):
        if p in ("a", "aː"):
            window = out[max(0, idx - 2): idx + 3]
            if any(w in EMPHATICS_IPA or w == "q" for w in window):
                out[idx] = "ɑː" if p == "aː" else "ɑ"
    return out


def _epenthesis(phones: List[str]) -> List[str]:
    """Break an unpronounceable CCC sequence with an epenthetic /e/ (Levantine repair)."""
    out: List[str] = []
    run = 0
    for p in phones:
        is_cons = p in CONSONANT_IPA
        if is_cons:
            run += 1
            if run >= 3:
                out.insert(len(out), "e")
                run = 1
        else:
            run = 0
        out.append(p)
    return out


def levantine_g2p(text: str, back_emphatics: bool = True, epenthesize: bool = True) -> str:
    """Convert *diacritised* Levantine Arabic text to an IPA string."""
    words = text.split()
    rendered: List[str] = []
    for w in words:
        phones = _g2p_word(w)
        if back_emphatics:
            phones = _emphatic_backing(phones)
        if epenthesize:
            phones = _epenthesis(phones)
        rendered.append("".join(phones))
    return " ".join(r for r in rendered if r)


# ----------------------------------------------------------------------------------
# Fallback path: espeak-ng MSA IPA, surface-remapped toward Levantine.
# Used only when the input is *undiacritised* and no diacritiser backend is installed.
# Lower quality than the rule G2P on diacritised text, but it runs anywhere and still
# carries the two headline dialect features (ق→ʔ, ج→ʒ).
# ----------------------------------------------------------------------------------
_MSA_TO_LEV = [
    ("d͡ʒ", "ʒ"), ("dʒ", "ʒ"),   # ج  affricate -> Levantine /ʒ/
    ("q", "ʔ"),                   # ق  -> glottal stop
    ("ðˤ", "zˤ"), ("ð", "d"), ("θ", "t"),  # interdental mergers
]


def msa_ipa_to_levantine(ipa: str) -> str:
    for a, b in _MSA_TO_LEV:
        ipa = ipa.replace(a, b)
    return ipa


def arabic_fallback_ipa(text: str, back_emphatics: bool = True) -> str:
    """Phonemise *undiacritised* Arabic via espeak-ng, then remap toward Levantine."""
    from . import espeak
    from .phoneme_inventory import fold_to_inventory, tokenize_ipa

    if not espeak.available():
        # last resort: strip to consonants via the rule engine (intelligible-ish)
        return levantine_g2p(text, back_emphatics=back_emphatics)
    raw = espeak.phonemize(text, voice="ar")
    raw = msa_ipa_to_levantine(raw)
    out_words = []
    for word in raw.split(" "):
        toks = tokenize_ipa(word).symbols
        out_words.append("".join(fold_to_inventory(t) for t in toks))
    ipa = " ".join(w for w in out_words if w)
    return ipa


if __name__ == "__main__":
    tests = [
        ("بَيت", "beːt (house)"),
        ("يَوم", "joːm (day)"),
        ("قَلْب", "ʔalb (heart)"),
        ("جِبْنة", "ʒibne (cheese)"),
        ("صار", "sˤaːr -> backed (became)"),
        ("الشَّمس", "iʃʃams (the sun, sun-letter)"),
        ("القَمَر", "ilʔamar (the moon, moon-letter + ق->ʔ)"),
        ("مَرحَبا", "marħaba (hello)"),
        ("الله", "ʔaɫɫa (lexicon)"),
        ("قَديش", "ʔadeːʃ (how much, lexicon-ish)"),
    ]
    for ar, gloss in tests:
        print(f"{ar:>10}  ->  {levantine_g2p(ar):<16}  [{gloss}]")
