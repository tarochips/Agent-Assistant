def test_health_endpoints(client) -> None:
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_session_crud(client) -> None:
    created = client.post("/api/sessions", json={"title": "测试会话"})
    assert created.status_code == 201
    session_id = created.json()["id"]

    detail = client.get(f"/api/sessions/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["title"] == "测试会话"

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True}


def test_upload_and_stream_chat_with_citations(client) -> None:
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("knowledge.md", "RAG通过检索资料增强大模型回答。", "text/markdown")},
    )
    assert upload.status_code == 201
    assert upload.json()["chunk_count"] == 1

    session = client.post("/api/sessions").json()
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "RAG如何增强回答？", "session_id": session["id"]},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: retrieval" in body
    assert '"filename":"knowledge.md"' in body
    assert "event: token" in body
    assert "event: completed" in body

    saved = client.get(f"/api/sessions/{session['id']}").json()
    assert len(saved["messages"]) == 2
    assert saved["messages"][1]["content"] == "测试回答"
    assert saved["messages"][1]["sources"][0]["filename"] == "knowledge.md"


def test_invalid_chat_request_is_rejected(client) -> None:
    response = client.post("/api/chat/stream", json={"message": "   "})
    assert response.status_code == 422


def test_unsupported_document_is_rejected(client) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("document.pdf", b"not-a-pdf", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_DOCUMENT"
