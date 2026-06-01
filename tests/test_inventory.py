from hams_tts.text import phoneme_inventory as inv


def test_vocab_is_stable_and_unique():
    assert inv.VOCAB_SIZE == len(set(inv.SYMBOLS))
    # specials occupy the first slots (frozen contract for the model embedding)
    assert inv.SYMBOLS[0] == inv.PAD
    assert inv.SYMBOL_TO_ID[inv.BOS] >= 0


def test_longest_match_tokenisation_of_multichar_symbols():
    # tie-bar, length, pharyngealisation must tokenise as single units
    toks = inv.tokenize_ipa("t͡ʃaːsˤ").symbols
    assert "t͡ʃ" in toks and "aː" in toks and "sˤ" in toks
    assert "t" not in toks  # the bare /t/ must NOT appear (was consumed by t͡ʃ)


def test_encode_decode_roundtrip():
    ids, syms = inv.encode("ʔadeːʃ", add_bos_eos=True)
    assert syms[0] == inv.BOS and syms[-1] == inv.EOS
    assert inv.decode(ids) == "ʔadeːʃ"


def test_unknowns_are_folded_never_dropped():
    # ascii 'g' folds to IPA ɡ; an exotic char becomes UNK but is retained
    toks = inv.tokenize_ipa("gƭ").symbols  # ƭ has no mapping
    assert toks[0] == "ɡ"
    assert inv.UNK in toks
