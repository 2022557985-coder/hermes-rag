"""Dataset loader for evaluation."""

import json
from pathlib import Path
from typing import Any


class EvaluationDataset:
    """Load and manage evaluation datasets."""

    def __init__(self, data_path: str = ""):
        if not data_path:
            data_path = str(Path(__file__).parent / "data" / "ground_truth.json")
        self.data_path = data_path
        self._data: list[dict[str, Any]] = []

    def load(self) -> list[dict[str, Any]]:
        """Load the ground truth dataset.

        Returns:
            List of dicts with keys: query, relevant_chunk_ids, category.
        """
        try:
            with open(self.data_path, encoding="utf-8") as f:
                self._data = json.load(f)
        except FileNotFoundError:
            self._data = self._generate_sample_data()
        return self._data

    def _generate_sample_data(self) -> list[dict[str, Any]]:
        """Generate a sample evaluation dataset for testing."""
        return [
            {
                "query": "什么是机器学习？",
                "relevant_chunk_ids": ["ml_intro.md_0"],
                "category": "ml",
            },
            {
                "query": "监督学习有哪些类型？",
                "relevant_chunk_ids": ["ml_supervised.md_0", "ml_supervised.md_1"],
                "category": "ml",
            },
            {
                "query": "什么是神经网络？",
                "relevant_chunk_ids": ["ml_neural.md_0"],
                "category": "ml",
            },
            {
                "query": "如何评估模型性能？",
                "relevant_chunk_ids": ["ml_eval.md_0"],
                "category": "ml",
            },
            {
                "query": "K-means是什么算法？",
                "relevant_chunk_ids": ["ml_clustering.md_0"],
                "category": "ml",
            },
            {
                "query": "Python的特点是什么？",
                "relevant_chunk_ids": ["py_basics.md_0"],
                "category": "python",
            },
            {
                "query": "Python支持哪些编程范式？",
                "relevant_chunk_ids": ["py_features.md_0"],
                "category": "python",
            },
            {
                "query": "分类和回归有什么区别？",
                "relevant_chunk_ids": ["ml_classification.md_0", "ml_regression.md_0"],
                "category": "ml",
            },
            {
                "query": "CNN是什么架构？",
                "relevant_chunk_ids": ["ml_arch.md_0"],
                "category": "ml",
            },
            {
                "query": "交叉验证的作用是什么？",
                "relevant_chunk_ids": ["ml_cv.md_0"],
                "category": "ml",
            },
        ]

    def get_queries(self) -> list[str]:
        """Get all queries from the dataset."""
        if not self._data:
            self.load()
        return [item["query"] for item in self._data]

    def get_relevant_ids(self, query: str) -> list[str]:
        """Get relevant chunk IDs for a query."""
        if not self._data:
            self.load()
        for item in self._data:
            if item["query"] == query:
                return item["relevant_chunk_ids"]
        return []

    def size(self) -> int:
        """Return the number of queries."""
        if not self._data:
            self.load()
        return len(self._data)