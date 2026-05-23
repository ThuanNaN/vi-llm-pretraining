"""Unit tests for MinHash deduplication helpers."""

from vi_llm.dataprep.dedup import _make_minhash, _shingles


def test_shingles_basic():
    s = _shingles("hello world", n=5)
    assert "hello" in s
    assert "world" in s


def test_shingles_short_text():
    # text shorter than n should still produce at least one shingle
    s = _shingles("hi", n=5)
    assert len(s) >= 1


def test_minhash_identical_texts():
    m1 = _make_minhash("Xin chào Việt Nam hôm nay trời đẹp")
    m2 = _make_minhash("Xin chào Việt Nam hôm nay trời đẹp")
    assert m1.jaccard(m2) == pytest.approx(1.0, abs=0.01)


def test_minhash_different_texts():
    m1 = _make_minhash("Xin chào Việt Nam")
    m2 = _make_minhash("Python là ngôn ngữ lập trình phổ biến")
    assert m1.jaccard(m2) < 0.3


def test_minhash_near_duplicate():
    base = "Hôm nay trời rất đẹp và nắng ấm áp"
    near = base + " thêm một chữ"
    m1 = _make_minhash(base)
    m2 = _make_minhash(near)
    assert m1.jaccard(m2) > 0.5


import pytest
