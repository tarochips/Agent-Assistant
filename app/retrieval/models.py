from dataclasses import dataclass

from app.schemas.events import Citation


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float

    def to_citation(self, excerpt_length: int = 180) -> Citation:
        excerpt = self.text.strip().replace("\n", " ")
        if len(excerpt) > excerpt_length:
            excerpt = excerpt[:excerpt_length].rstrip() + "..."
        return Citation(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            filename=self.filename,
            chunk_index=self.chunk_index,
            score=round(self.score, 6),
            excerpt=excerpt,
        )
