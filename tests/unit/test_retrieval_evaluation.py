import pytest

from app.retrieval.evaluation import retrieval_metrics


def test_retrieval_metrics_calculate_recall_mrr_and_hit_rate() -> None:
    metrics = retrieval_metrics(
        [["a:0", "b:0", "c:0"], ["d:0", "e:0"]],
        [{"b:0", "c:0"}, {"missing:0"}],
        k=2,
    )

    assert metrics == {
        "recall@2": 0.25,
        "mrr@2": 0.25,
        "hit_rate@2": 0.5,
        "queries": 2,
    }


def test_retrieval_metrics_validate_input() -> None:
    with pytest.raises(ValueError, match="corresponding"):
        retrieval_metrics([["a:0"]], [], k=1)
    with pytest.raises(ValueError, match="positive"):
        retrieval_metrics([], [], k=0)
