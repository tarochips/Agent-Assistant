import pytest

from app.retrieval.tfidf import split_text


def test_split_text_uses_overlap() -> None:
    chunks = split_text("abcdefghij", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_split_text_returns_empty_for_whitespace() -> None:
    assert split_text("   \n", chunk_size=4, overlap=1) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (4, -1), (4, 4), (4, 5)],
)
def test_split_text_rejects_invalid_window(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("content", chunk_size=chunk_size, overlap=overlap)
