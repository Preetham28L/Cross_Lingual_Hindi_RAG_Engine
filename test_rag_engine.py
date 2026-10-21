import unicodedata
from rag_engine import normalize_text, split_paragraphs


def test_normalize_preserves_hindi_danda():
    text = "कलाम   महान थे।\n\n\nभारत"
    cleaned = normalize_text(text)
    assert "।" in cleaned
    assert "\n\n\n" not in cleaned


def test_unicode_nfc():
    sample = "हिंदी"
    assert normalize_text(sample) == unicodedata.normalize("NFC", sample)


def test_paragraph_split_keeps_content():
    parts = split_paragraphs("पहला वाक्य।\n\nदूसरा वाक्य।")
    assert any("पहला" in p for p in parts)
    assert any("दूसरा" in p for p in parts)
