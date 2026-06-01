"""Tests pin the *Levantine signatures* the assessment explicitly asks about."""

from hams_tts.text.levantine_g2p import levantine_g2p, msa_ipa_to_levantine


def test_qaf_becomes_glottal_stop():
    # ق -> ʔ  (headline Levantine rule)
    assert levantine_g2p("قَلْب") == "ʔalb"
    assert levantine_g2p("القَمَر") == "ʔilʔamar"


def test_jim_becomes_zh():
    # ج -> ʒ  (not the MSA affricate /d͡ʒ/)
    assert levantine_g2p("جِبْنة") == "ʒibne"
    assert "ʒ" in levantine_g2p("جَمَل")


def test_diphthong_monophthongisation():
    # ay -> eː , aw -> oː
    assert levantine_g2p("بَيت") == "beːt"
    assert levantine_g2p("يَوم") == "joːm"


def test_teh_marbuta_imala_to_e():
    assert levantine_g2p("جِبْنة").endswith("e")


def test_sun_letter_assimilation_geminates():
    out = levantine_g2p("الشَّمس")
    assert out.startswith("ʔiʃʃ")  # lam assimilates -> /ʃ/ geminate


def test_moon_letter_keeps_lam():
    assert levantine_g2p("القَمَر").startswith("ʔil")


def test_emphatic_backing():
    # /a/ near ص backs to ɑ
    assert "ɑ" in levantine_g2p("صار")


def test_lexicon_overrides():
    assert levantine_g2p("الله") == "ʔaɫɫa"
    assert levantine_g2p("شو") == "ʃuː"


def test_msa_to_levantine_surface_remap():
    assert msa_ipa_to_levantine("qalb") == "ʔalb"
    assert msa_ipa_to_levantine("d͡ʒamal") == "ʒamal"
