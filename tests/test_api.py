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


def test_validation_and_unknown_resources():
    with TestClient(app) as client:
        assert client.post("/api/ask", json={"question": "hi", "strategy": "x"}).status_code == 422
        assert client.get("/api/pdf/nope_FY99").status_code == 404
        assert client.get("/api/eval/..%2Fsecret").status_code == 404
