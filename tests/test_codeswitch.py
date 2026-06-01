from hams_tts.text.codeswitch import segment


def test_pure_arabic_single_span():
    spans = segment("مرحبا كيف حالك؟")
    assert len(spans) == 1 and spans[0].lang == "ar"


def test_pure_english_single_span():
    spans = segment("hello how are you")
    assert len(spans) == 1 and spans[0].lang == "en"


def test_basic_codeswitch_alternation():
    spans = segment("عندي meeting بكرا")
    langs = [s.lang for s in spans]
    assert langs == ["ar", "en", "ar"]


def test_number_attaches_to_surrounding_language():
    # the digit sits inside an Arabic context -> stays Arabic
    spans = segment("الساعة 3:30 بعد الظهر")
    assert len(spans) == 1 and spans[0].lang == "ar"
    assert "3:30" in spans[0].text


def test_english_number_stays_english():
    spans = segment("meeting at 3:30 pm")
    assert all(s.lang == "en" for s in spans)


def test_leading_neutral_resolves_to_following_language():
    spans = segment("123 مرحبا")
    assert spans[0].lang == "ar"  # digits resolve forward to Arabic
