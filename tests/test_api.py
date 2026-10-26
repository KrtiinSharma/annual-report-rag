from fastapi.testclient import TestClient

from annual_report_rag.api.main import app


def test_reports_and_health_work_without_an_index():
    with TestClient(app) as client:
        r = client.get("/api/reports")
        assert r.status_code == 200
        body = r.json()
        assert {c["slug"] for c in body["companies"]} >= {"indigo", "infosys", "tcs"}
        assert body["examples"]
        assert client.get("/api/health").json()["status"] == "ok"
        root = client.get("/", follow_redirects=False)
        assert root.status_code == 307 and root.headers["location"] == "/docs"


def test_pipeline_failures_become_json_errors_with_cors(monkeypatch):
    import httpx
    from groq import RateLimitError

    import annual_report_rag.rag.chain as chain

    def rate_limited(*args):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
        raise RateLimitError("tokens per day", response=response, body=None)

    origin = {"Origin": "http://localhost:5173"}
    with TestClient(app) as client:
        monkeypatch.setattr(chain, "answer_question", rate_limited)
        r = client.post("/api/ask", json={"question": "What was revenue?"}, headers=origin)
        assert r.status_code == 503 and "quota" in r.json()["detail"]
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"

        monkeypatch.setattr(chain, "answer_question", lambda *a: 1 / 0)
        r = client.post("/api/ask", json={"question": "What was revenue?"}, headers=origin)
        assert r.status_code == 500 and r.json()["detail"]


def test_answer_model_falls_back_to_other_models():
    from annual_report_rag.retrieval.models import get_chat_model

    llm = get_chat_model("answer")
    assert [m.model_name for m in [llm.runnable, *llm.fallbacks]] == [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]


def test_validation_and_unknown_resources():
    with TestClient(app) as client:
        assert client.post("/api/ask", json={"question": "hi", "strategy": "x"}).status_code == 422
        assert client.get("/api/pdf/nope_FY99").status_code == 404
        assert client.get("/api/eval/..%2Fsecret").status_code == 404
