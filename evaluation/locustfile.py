"""Locust stress test for Hermes-RAG API."""

from locust import HttpUser, between, task


class HermesRAGUser(HttpUser):
    """Simulated user for Hermes-RAG API stress testing."""

    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Check health on start."""
        with self.client.get("/api/v1/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")

    @task(3)
    def query_ml(self):
        """Query about machine learning."""
        self.client.post(
            "/api/v1/query",
            json={
                "query": "什么是机器学习？",
                "top_k": 5,
                "use_reranker": True,
                "generate_answer": False,
            },
            timeout=10,
        )

    @task(2)
    def query_python(self):
        """Query about Python."""
        self.client.post(
            "/api/v1/query",
            json={
                "query": "Python的特点是什么？",
                "top_k": 3,
                "use_reranker": True,
                "generate_answer": False,
            },
            timeout=10,
        )

    @task(1)
    def health_check(self):
        """Health check."""
        self.client.get("/api/v1/health")