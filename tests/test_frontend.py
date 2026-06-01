from hams_tts.text.frontend import TextFrontend
from hams_tts.text.phoneme_inventory import BOS, EOS, VOCAB_SIZE, Lang


def _fe():
    return TextFrontend()


def test_streams_are_aligned_and_wrapped():
    u = _fe().process("مَرحَبا")
    assert u.symbols[0] == BOS and u.symbols[-1] == EOS
    assert len(u.phoneme_ids) == len(u.language_ids) == len(u.symbols)


def test_all_phoneme_ids_in_range():
    u = _fe().process("بَدّي إحجِز flight بُكرا")
    assert all(0 <= i < VOCAB_SIZE for i in u.phoneme_ids)


def test_codeswitch_produces_both_language_ids():
    u = _fe().process("Hams AI بَتِشتِغِل real-time")
    assert int(Lang.AR) in u.language_ids
    assert int(Lang.EN) in u.language_ids


def test_diacritized_arabic_uses_rule_g2p_signatures():
    # fully diacritised -> high-quality rule path -> ق becomes ʔ
    u = _fe().process("قَلْبي")
    assert "ʔ" in u.ipa


def test_pure_english_has_no_arabic_lang_ids():
    u = _fe().process("real time streaming")
    assert int(Lang.AR) not in u.language_ids
    assert int(Lang.EN) in u.language_ids


def test_empty_input_is_safe():
    u = _fe().process("")
    assert u.symbols == [BOS, EOS]
