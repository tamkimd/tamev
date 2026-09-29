# Contributing to TAMEV

First off, thank you for considering contributing to TAMEV! Real-time System One decision models thrive with community feedback, new architectural encoders, and improved edge deployment runtimes.

---

## 🛠️ Development Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) for fast, isolated Python dependency management:

```bash
# 1. Enter the TAMEV source checkout and create a virtual environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Install pre-commit hooks
pre-commit install
```

---

## 🧪 Running Tests

TAMEV maintains a strict test suite covering option-label bookkeeping under permutation, multi-format export, and TypeSafe protocol compatibility:

```bash
# Run the complete test suite
pytest tests/ -v

# Run the permutation invariance / option-label bookkeeping tests
pytest tests/test_architecture.py -v
```

---

## 🎨 Code Style & Quality

We enforce strict formatting and linting rules using [Ruff](https://astral.sh/ruff):

```bash
# Lint check and auto-fix
ruff check . --fix

# Format code
ruff format .

# Run pre-commit across all files
pre-commit run --all-files
```

---

## 🚀 Submitting a Pull Request

1. **Fork the repo** and create your branch from `main`:
   ```bash
   git checkout -b feature/my-cool-feature
   ```
2. **Write tests** for any new feature or bug fix.
3. Ensure all tests and linter checks pass locally:
   ```bash
   pytest tests/
   ruff check .
   ```
4. **Commit your changes** with descriptive commit messages following Conventional Commits (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
5. Push to your fork and submit a Pull Request to `main`.

Thank you for helping make TAMEV faster and more reliable!
