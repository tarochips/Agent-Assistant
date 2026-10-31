"""Offline, deterministic citation checks against manually labeled evidence IDs."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class CitationLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1)


class ReportCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report: str = Field(min_length=1)
    citations: list[CitationLabel]
    relevant_chunk_ids: list[str] = Field(min_length=1)


def evaluate_reports(cases: list[ReportCase]) -> dict[str, Any]:
    cited = valid = covered = has_citation = 0
    for case in cases:
        references = [int(n) for n in re.findall(r"\[S(\d+)\]", case.report)]
        cited += len(references)
        has_citation += bool(references)
        valid_refs = [n for n in references if 1 <= n <= len(case.citations)]
        valid += len(valid_refs)
        used = {case.citations[n - 1].chunk_id for n in valid_refs}
        covered += bool(used & set(case.relevant_chunk_ids))
    count = len(cases)
    return {
        "reports": count,
        "citation_references": cited,
        "reports_with_citations": has_citation,
        "valid_reference_rate": valid / cited if cited else 0.0,
        "labeled_evidence_hit_rate": covered / count if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate report citation structure (not factual correctness)."
    )
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = TypeAdapter(list[ReportCase]).validate_python(data)
    print(json.dumps(evaluate_reports(cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
