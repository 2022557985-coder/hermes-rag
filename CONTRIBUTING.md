# Contributing

Thank you for contributing to Hermes-RAG.

## Getting started

1. Fork the repository and clone it locally.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the test suite: `pytest`.
4. Run the canonical benchmark: `python run_eval.py`.

## Code style

- Keep functions and classes focused; prefer small modules.
- Use type hints for public APIs.
- Add a test for every bug fix.
- Run `ruff check .` before submitting a pull request.

## Pull request checklist

- [ ] Tests pass locally.
- [ ] New behavior is covered by tests.
- [ ] Documentation is updated when behavior changes.
- [ ] The benchmark report is regenerated when retrieval behavior changes.