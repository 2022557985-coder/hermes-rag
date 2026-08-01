"""Pytest fixtures for Hermes-RAG tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_docs_dir():
    """Return path to the sample documents directory."""
    samples_dir = Path(__file__).parent.parent / "evaluation" / "data" / "sample_docs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return str(samples_dir)


@pytest.fixture
def sample_text():
    """Return sample text for testing."""
    return """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn from data.
It focuses on the development of algorithms that can improve through experience.

## Supervised Learning

Supervised learning is the most common type of machine learning.
It involves training a model on labeled data, where the correct output is known.

### Classification

Classification is a supervised learning task where the output is a category.
Examples include spam detection and image recognition.

### Regression

Regression is a supervised learning task where the output is a continuous value.
Examples include house price prediction and stock market forecasting.

## Unsupervised Learning

Unsupervised learning involves finding patterns in unlabeled data.
Common techniques include clustering and dimensionality reduction.

### Clustering

Clustering groups similar data points together.
K-means and hierarchical clustering are popular algorithms.

## Neural Networks

Neural networks are computing systems inspired by biological neural networks.
Deep learning uses neural networks with many layers.

### Architecture

Common architectures include:
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Transformer models

## Model Evaluation

Model evaluation is critical for assessing performance.
Metrics include accuracy, precision, recall, and F1-score.

### Cross-Validation

Cross-validation helps assess how well a model generalizes.
K-fold cross-validation is the most common approach.
"""


@pytest.fixture
def sample_chunks():
    """Return sample chunks for testing."""
    return [
        {
            "chunk_id": "doc1_0",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"source": "doc1.txt", "heading_path": "Introduction to Machine Learning"},
        },
        {
            "chunk_id": "doc1_1",
            "text": "Supervised learning is the most common type of machine learning. It involves training a model on labeled data.",
            "metadata": {"source": "doc1.txt", "heading_path": "Introduction to Machine Learning > Supervised Learning"},
        },
        {
            "chunk_id": "doc1_2",
            "text": "Neural networks are computing systems inspired by biological neural networks. Deep learning uses neural networks with many layers.",
            "metadata": {"source": "doc1.txt", "heading_path": "Introduction to Machine Learning > Neural Networks"},
        },
        {
            "chunk_id": "doc2_0",
            "text": "Python is a high-level programming language known for its readability and simplicity.",
            "metadata": {"source": "doc2.txt", "heading_path": "Python Basics"},
        },
        {
            "chunk_id": "doc2_1",
            "text": "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "metadata": {"source": "doc2.txt", "heading_path": "Python Basics > Features"},
        },
    ]


@pytest.fixture
def sample_config():
    """Return a sample config dict."""
    return {
        "dense_top_k": 100,
        "sparse_top_k": 100,
        "fusion_top_k": 50,
        "reranking": {
            "enabled": True,
            "timeout_seconds": 1.5,
        },
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir