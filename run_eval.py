#!/usr/bin/env python3
"""Canonical evaluation runner for Hermes-RAG.

Builds a fresh temporary index, evaluates retrieval variants, and writes a
reproducible JSON report (default: docs/eval_results.json).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Hermes-RAG canonical benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "eval_results.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "rrf", "rerank"],
        choices=["baseline", "rrf", "rerank", "cross"],
        help="Variants to evaluate",
    )
    parser.add_argument(
        "--extra-docs",
        action="store_true",
        help="Include every sample document (not just the 11 benchmark docs)",
    )
    args = parser.parse_args()

    from evaluation.benchmark import print_benchmark_report, run_benchmark

    results = run_benchmark(
        output_path=args.output,
        variants=args.variants,
        extra_docs=args.extra_docs,
    )
    print_benchmark_report(results)


if __name__ == "__main__":
    main()