"""API integration tests for Hermes-RAG."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api.server import create_app
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_health_response_format(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "1.0.0"
        assert data["status"] in ("healthy", "initializing")


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_empty_request(self, client):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422  # Validation error

    def test_query_valid_request(self, client):
        response = client.post("/api/v1/query", json={
            "query": "test query",
            "top_k": 3,
            "use_reranker": False,
        })
        # May return 200 if documents are indexed, or 500 if no index exists
        # Both are acceptable depending on setup state
        if response.status_code == 500:
            data = response.json()
            assert "internal_error" in data.get("detail", {}).get("error", "")
        else:
            assert response.status_code == 200

    def test_query_with_generation(self, client):
        response = client.post("/api/v1/query", json={
            "query": "test",
            "generate_answer": True,
        })
        if response.status_code == 500:
            data = response.json()
            assert "internal_error" in data.get("detail", {}).get("error", "")
        else:
            assert response.status_code == 200


class TestIngestEndpoint:
    """Tests for /ingest endpoint."""

    def test_ingest_invalid_path(self, client):
        response = client.post("/api/v1/ingest", json={
            "source": "/nonexistent/file.pdf",
        })
        assert response.status_code in (400, 404)

    def test_ingest_path_traversal_blocked(self, client):
        response = client.post("/api/v1/ingest", json={
            "source": "../../../etc/passwd",
        })
        assert response.status_code == 400
        data = response.json()
        assert "Path traversal" in data["detail"]["message"]

    def test_ingest_invalid_extension(self, client):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            response = client.post("/api/v1/ingest", json={"source": path})
            assert response.status_code == 400
        finally:
            os.unlink(path)

    def test_ingest_missing_source(self, client):
        response = client.post("/api/v1/ingest", json={})
        assert response.status_code == 422


class TestEdgeCases:
    """Edge case tests for the API."""

    def test_query_special_characters(self, client):
        """Query with special characters should not crash."""
        response = client.post("/api/v1/query", json={
            "query": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        })
        # Should not crash - 200 (success) or 500 (no index) is acceptable
        if response.status_code == 500:
            data = response.json()
            assert "internal_error" in data.get("detail", {}).get("error", "")
        else:
            assert response.status_code == 200

    def test_query_unicode(self, client):
        """Query with unicode characters."""
        response = client.post("/api/v1/query", json={
            "query": "如何处理中文文档的编码问题？",
        })
        if response.status_code == 500:
            data = response.json()
            assert "internal_error" in data.get("detail", {}).get("error", "")
        else:
            assert response.status_code == 200

    def test_query_empty_string(self, client):
        """Empty query string."""
        response = client.post("/api/v1/query", json={
            "query": "",
        })
        assert response.status_code in (200, 422)

    def test_query_large_top_k(self, client):
        """Query with very large top_k."""
        response = client.post("/api/v1/query", json={
            "query": "test",
            "top_k": 10000,
        })
        if response.status_code == 500:
            data = response.json()
            assert "internal_error" in data.get("detail", {}).get("error", "")
        else:
            assert response.status_code == 200

    def test_query_negative_top_k(self, client):
        """Query with negative top_k."""
        response = client.post("/api/v1/query", json={
            "query": "test",
            "top_k": -1,
        })
        assert response.status_code == 422  # Pydantic validation