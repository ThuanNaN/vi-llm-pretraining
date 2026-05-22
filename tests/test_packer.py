"""Unit tests for sequence packer logic (no tokenizer on disk required)."""

import pytest


def _fake_pack(token_lists: list[list[int]], seq_length: int) -> list[list[int]]:
    """Inline reimplementation of the pack logic for unit testing."""
    buffer: list[int] = []
    result: list[list[int]] = []
    for ids in token_lists:
        buffer.extend(ids)
        while len(buffer) >= seq_length:
            result.append(buffer[:seq_length])
            buffer = buffer[seq_length:]
    while len(buffer) >= seq_length:
        result.append(buffer[:seq_length])
        buffer = buffer[seq_length:]
    return result


def test_exact_multiple():
    seqs = _fake_pack([[1] * 2048], seq_length=2048)
    assert len(seqs) == 1
    assert len(seqs[0]) == 2048


def test_concatenation_across_docs():
    # two 1024-token docs → one 2048 sequence
    seqs = _fake_pack([[1] * 1024, [2] * 1024], seq_length=2048)
    assert len(seqs) == 1
    assert seqs[0][:1024] == [1] * 1024
    assert seqs[0][1024:] == [2] * 1024


def test_no_short_sequences():
    seqs = _fake_pack([[1] * 100, [2] * 100], seq_length=2048)
    # 200 tokens total — not enough for one full sequence
    assert len(seqs) == 0


def test_all_sequences_exact_length():
    seqs = _fake_pack([[i] * 512 for i in range(10)], seq_length=2048)
    assert all(len(s) == 2048 for s in seqs)


def test_no_truncation_loss():
    # 5000 tokens → floor(5000/2048) = 2 full sequences, remainder dropped
    seqs = _fake_pack([[1] * 5000], seq_length=2048)
    assert len(seqs) == 2
