from hams_tts.text import normalize as N


def test_arabic_indic_digits_and_levantine_cardinals():
    assert N.arabic_int_to_words(3) == "تْلاتة"
    assert N.arabic_int_to_words(2) == "تْنين"
    # Arabic-Indic digit normalisation then verbalisation
    out = N.normalize("عندي ٣ كتب", "ar")
    assert "تْلاتة" in out


def test_english_numbers_and_currency():
    out = N.normalize("it costs $5.50", "en")
    assert "five dollars" in out and "fifty cents" in out


def test_percent_and_time_both_languages():
    assert "percent" in N.normalize("95%", "en")
    assert "بالمية" in N.normalize("95%", "ar")
    assert "three thirty" in N.normalize("3:30", "en")


def test_unicode_normalisation_preserves_hamza():
    # conservative: hamza-carrying alef must survive (phonemic glottal stop)
    s = N.normalize_unicode("أكل إجا آدم")
    assert "أ" in s and "إ" in s and "آ" in s


def test_punctuation_canonicalisation():
    s = N.normalize_unicode("مرحبا، كيف؟")
    assert "،" not in s and "؟" not in s
    assert "," in s and "?" in s
