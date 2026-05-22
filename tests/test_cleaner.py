"""Unit tests for text cleaner (no fasttext — language_id disabled)."""

import pytest
from vi_llm.data.cleaner import CleanerConfig, TextCleaner


@pytest.fixture
def cleaner():
    cfg = CleanerConfig(language_id_enabled=False)
    return TextCleaner(cfg)


def test_strips_html(cleaner):
    result = cleaner.transform("<p>Xin chào <b>Việt Nam</b></p>")
    assert "<" not in result
    assert "Xin chào" in result


def test_normalizes_unicode(cleaner):
    # composed vs decomposed form of ộ
    composed = "lội"
    decomposed = "lội"
    assert cleaner.transform(decomposed) == cleaner.transform(composed)


def test_length_filter_too_short(cleaner):
    assert cleaner.clean("Hi") is None


def test_length_filter_too_long(cleaner):
    assert cleaner.clean("a" * 200_000) is None


def test_non_alphanum_ratio(cleaner):
    # mostly punctuation
    bad = "!@#$%^&*()" * 10 + "a"
    assert cleaner.clean(bad) is None


def test_repeated_char_ratio(cleaner):
    bad = "a" * 90 + "bcd efgh"  # 'a' is >80% of chars
    assert cleaner.clean(bad) is None


def test_valid_vietnamese_text(cleaner):
    text = "Hôm nay trời đẹp, chúng ta cùng đi học. " * 5
    result = cleaner.clean(text)
    assert result is not None
    assert "Hôm nay" in result
