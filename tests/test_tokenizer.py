"""Unit tests for tokenizer training helpers (no file I/O)."""

import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def _build_tiny_tokenizer() -> Tokenizer:
    """Train a tiny BPE tokenizer on a small Vietnamese corpus for testing."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    corpus = [
        "Xin chào Việt Nam",
        "Hôm nay trời đẹp",
        "Lập trình viên Python",
        "Học máy và trí tuệ nhân tạo",
        "Ngôn ngữ lớn được huấn luyện trước",
    ] * 20  # repeat to get enough frequency

    special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        special_tokens=special_tokens,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok


@pytest.fixture(scope="module")
def tokenizer():
    return _build_tiny_tokenizer()


def test_encodes_vietnamese_diacritics(tokenizer):
    text = "Xin chào Việt Nam"
    enc = tokenizer.encode(text)
    assert len(enc.ids) > 0
    decoded = tokenizer.decode(enc.ids)
    assert decoded == text


def test_special_tokens_present(tokenizer):
    vocab = tokenizer.get_vocab()
    for tok in ["<s>", "</s>", "<pad>", "<unk>"]:
        assert tok in vocab, f"Missing special token: {tok}"


def test_round_trip(tokenizer):
    texts = [
        "Hôm nay trời rất đẹp",
        "Lập trình bằng Python rất thú vị",
        "Học máy là lĩnh vực phát triển nhanh",
    ]
    for text in texts:
        ids = tokenizer.encode(text).ids
        assert tokenizer.decode(ids) == text


def test_no_empty_encoding(tokenizer):
    enc = tokenizer.encode("abc")
    assert len(enc.ids) > 0
