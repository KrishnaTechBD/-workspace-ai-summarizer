# AI Text Summarizer (CPU-Only)

A lightweight, production-ready text-summarization tool powered by
[Hugging Face Transformers](https://huggingface.co/docs/transformers),
optimized to run on **CPU only** — no GPU, no CUDA dependencies. Designed
for use in CI/CD pipelines, local development, and lightweight production
deployments.

---

## Overview

- **CPU-only inference** — works on any machine, including GitHub-hosted
  runners (`ubuntu-latest`).
- **Hugging Face Transformers** — uses pre-trained summarization models from
  the Hub (default: [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)).
- **Tiny footprint** — pinned dependencies, fast install, minimal setup.
- **CI/CD ready** — ships with a GitHub Actions workflow that lints, tests,
  and runs an end-to-end smoke test.
- **Reproducible** — every dependency is pinned to an exact version.

## Quick Start

### 1. Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install everything — CPU-only PyTorch is pulled from the official index.
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Run the summarizer

```bash
# Use the built-in sample text (Apollo program)
python summarizer.py

# Summarize a custom string
python summarizer.py --text "Your long text here..."

# Summarize text from a file
python summarizer.py < article.txt

# Pipe text from another command
cat article.txt | python summarizer.py

# Use a different Hugging Face model
SUMMARIZER_MODEL=Falconsai/text_summarization python summarizer.py
```

### 3. Lint and test locally

```bash
flake8 .
pytest tests -v
```

## CI/CD Automation

A GitHub Actions workflow ships at `.github/workflows/ci.yml`. It runs on
every `push` and `pull_request` against `main` / `master`, across **Python
3.10 and 3.11**, and performs these steps:

1. **Checkout** the repository.
2. **Set up Python** with pip caching.
3. **Install PyTorch (CPU only)** via the official CPU index.
4. **Install remaining deps** from `requirements.txt`.
5. **Cache Hugging Face models** so subsequent runs skip the ~300 MB
   download.
6. **Lint** with `flake8` (syntax errors, complexity, line length).
7. **Run tests** with `pytest` — automatically skipped if `/tests` is
   absent.
8. **Smoke test** `summarizer.py` to verify end-to-end execution on CPU.

The full pipeline typically completes in **3–5 minutes** on a standard
GitHub runner. The first run downloads the default summarization model;
later runs hit the cache.

### Customizing the model in CI

Override the model without touching source code:

```yaml
env:
  SUMMARIZER_MODEL: Falconsai/text_summarization
```

### Adding tests

The CI runs `pytest tests/` whenever a `tests/` directory exists. Drop
any `test_*.py` file into that folder and it will be picked up
automatically — no extra wiring needed.

## Project Layout

```
.
├── summarizer.py            # CLI entry point
├── requirements.txt         # Pinned Python dependencies
├── .github/workflows/ci.yml # CI pipeline
├── .flake8                  # Lint config
├── tests/                   # Pytest test suite
│   ├── __init__.py
│   └── test_summarizer.py
└── README.md
```

## License

MIT
