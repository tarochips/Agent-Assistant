import asyncio

import pytest
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from app.api.dependencies import get_chat_service
from app.api.jobs import get_job_repository
from app.core.config import PROJECT_ROOT, Settings
from app.evaluation.reports import ReportCase, evaluate_reports
from app.jobs.worker import execute_claim
from app.main import create_app
from app.mcp_server.server import create_server
from app.retrieval.models import SearchResult


class MemoryJobs:
    def __init__(self):
        self.job = None
        self.completed = None
        self.failed = False
        self.lease_valid = True

    async def create(self, question):
        self.job = {"id": "job", "question": question, "status": "queued"}
        return self.job

    async def get(self, job_id):
        return self.job if job_id == "job" else None

    async def list_recent(self, limit):
        return [self.job] if self.job else []

    async def cancel(self, job_id):
        if not self.job or job_id != "job" or self.job["status"] != "queued":
            return False
        self.job["status"] = "cancelled"
        return True

    async def counts(self):
        return {self.job["status"]: 1} if self.job else {}

    async def heartbeat(self, job_id, token, lease_seconds):
        return self.lease_valid

    async def complete(self, job_id, token, result):
        if self.lease_valid:
            self.completed = result
            return True
        return False

    async def fail(self, job_id, token, *, attempts, max_attempts):
        self.failed = True
        return True


def test_job_api_create_status_metrics_cancel(client):
    repo = MemoryJobs()
    client.app.dependency_overrides[get_job_repository] = lambda: repo
    response = client.post("/api/research-jobs", json={"question": "  test  "})
    assert response.status_code == 202
    assert response.json()["question"] == "test"
    assert client.get("/api/research-jobs").json()[0]["id"] == "job"
    assert client.get("/api/research-jobs/job").json()["status"] == "queued"
    assert client.get("/api/research-jobs/metrics").json()["jobs_by_status"] == {"queued": 1}
    assert client.post("/api/research-jobs/job/cancel").status_code == 200
    assert client.post("/api/research-jobs/job/cancel").status_code == 409
    assert client.get("/api/research-jobs/missing").status_code == 404
    assert client.post("/api/research-jobs", json={"question": "   "}).status_code == 422


@pytest.mark.asyncio
async def test_job_worker_completes_and_fails(service_bundle):
    class Graph:
        async def ainvoke(self, data):
            return {"report": "result", "citations": [], "review": {"decision": "accept"}}

    repo = MemoryJobs()
    job = {"id": "job", "question": "test", "lease_token": "token", "attempts": 1}
    await execute_claim(repo, Graph(), job, service_bundle["settings"])
    assert repo.completed is not None
    assert repo.completed["report"] == "result"
    assert repo.completed["review"]["decision"] == "accept"

    class FailingGraph:
        async def ainvoke(self, data):
            raise RuntimeError("failure")

    await execute_claim(repo, FailingGraph(), job, service_bundle["settings"])
    assert repo.failed


@pytest.mark.asyncio
async def test_worker_cancels_execution_after_lease_loss(service_bundle):
    class BlockingGraph:
        def __init__(self):
            self.cancelled = False

        async def ainvoke(self, data):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    repo = MemoryJobs()
    repo.lease_valid = False
    graph = BlockingGraph()
    job = {"id": "job", "question": "test", "lease_token": "token", "attempts": 1}
    settings = service_bundle["settings"].model_copy(update={"job_lease_seconds": 0.03})
    await execute_claim(repo, graph, job, settings)
    assert graph.cancelled
    assert repo.completed is None
    assert not repo.failed


def test_api_token_protects_endpoints(monkeypatch, service_bundle, tmp_path):
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        static_dir=PROJECT_ROOT / "static",
        api_access_token="secret",
    )
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    application = create_app()
    application.dependency_overrides[get_chat_service] = lambda: service_bundle["chat_service"]
    application.dependency_overrides[get_job_repository] = lambda: MemoryJobs()
    with TestClient(application) as client:
        assert client.get("/api/health/live").status_code == 200
        assert client.get("/api/research-jobs/metrics").status_code == 401
        assert (
            client.get(
                "/api/research-jobs/metrics", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/api/research-jobs/metrics", headers={"Authorization": "Bearer secret"}
            ).status_code
            == 200
        )


@pytest.mark.asyncio
async def test_mcp_protocol_exposes_only_read_tools():
    class Retriever:
        async def search(self, query, *, top_k, min_score):
            return [SearchResult("doc:0", "doc", "doc.md", 0, "evidence", 0.8)]

    class Repository:
        async def list_documents(self):
            return [{"doc_id": "doc", "filename": "doc.md"}]

    server = create_server(Retriever(), Repository())
    async with create_connected_server_and_client_session(server) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {"search_knowledge", "list_documents"}
        result = await client.call_tool("search_knowledge", {"query": "sample", "top_k": 2})
        assert not result.isError
        assert "doc.md" in str(result.content)
        invalid = await client.call_tool("search_knowledge", {"query": "sample", "top_k": 100})
        assert invalid.isError


def test_report_evaluation_measures_only_citation_structure():
    cases = [
        ReportCase.model_validate(
            {
                "report": "statement [S1] and [S4]",
                "citations": [{"chunk_id": "a"}],
                "relevant_chunk_ids": ["a"],
            }
        )
    ]
    result = evaluate_reports(cases)
    assert result["valid_reference_rate"] == 0.5
    assert result["labeled_evidence_hit_rate"] == 1.0
