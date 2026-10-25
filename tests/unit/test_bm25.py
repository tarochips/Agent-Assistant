from app.retrieval.bm25 import BM25Retriever, tokenize
from app.retrieval.models import SearchResult


def make_result(chunk_id: str, text: str) -> SearchResult:
    document_id, chunk_index = chunk_id.split(":")
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.md",
        chunk_index=int(chunk_index),
        text=text,
        score=0.0,
    )


def test_tokenize_handles_chinese_and_english() -> None:
    tokens = tokenize("PostgreSQL 数据库存储知识")
    assert "postgresql" in tokens
    assert any(token in tokens for token in ["数据库", "数据"])


def test_bm25_ranks_exact_chinese_and_english_matches() -> None:
    docs = [
        make_result("1:0", "PostgreSQL 数据库使用 pgvector 存储向量。"),
        make_result("2:0", "花园里兔子喜欢吃胡萝卜。"),
        make_result("3:0", "使用 PostgreSQL 配置数据库连接。"),
    ]

    results = BM25Retriever().search("PostgreSQL 数据库连接", docs, top_k=2)

    assert [result.chunk_id for result in results] == ["3:0", "1:0"]
    assert results[0].score > results[1].score > 0.0


def test_bm25_returns_empty_for_query_without_matching_terms() -> None:
    docs = [make_result("1:0", "花园里兔子喜欢吃胡萝卜。")]
    assert BM25Retriever().search("不存在的主题", docs, top_k=5) == []
