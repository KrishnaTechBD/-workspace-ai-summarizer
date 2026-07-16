#!/usr/bin/env python3
"""
summarizer.py
=============
A lightweight, CPU-only text summarizer built on Hugging Face Transformers.

Designed to run on any machine, including GitHub-hosted runners
(no GPU, no CUDA dependencies). Defaults to the well-known
``sshleifer/distilbart-cnn-12-6`` model, but any compatible HF summarization
model can be selected via the ``SUMMARIZER_MODEL`` env var or ``--model`` flag.
"""
import argparse
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Force CPU mode BEFORE importing torch / transformers.
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("USE_CPU", "1")

# Silence noisy deprecation / FutureWarning noise from transformers internals.
warnings.filterwarnings("ignore")

import torch  # noqa: E402  (import after env-var setup, by design)
from transformers import pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get(
    "SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6"
)
# Stay well below the BART-family 1024-token limit to keep CPU inference fast.
MAX_INPUT_CHARS = 4000

# Built-in sample used when no input is supplied (CI smoke test).
SAMPLE_TEXT = (
    "The Apollo program was the third United States human spaceflight "
    "program carried out by NASA, which accomplished landing the first "
    "humans on the Moon from 1969 to 1972. First conceived during the "
    "Eisenhower administration as a three-man spacecraft to follow the "
    "one-man Project Mercury, Apollo was later dedicated to President "
    "Kennedy's national goal of landing a man on the Moon and returning "
    "him safely to Earth by the end of the 1960s. The Apollo program was "
    "launched by NASA in 1961 and used the Saturn family of rockets as "
    "launch vehicles. The Apollo spacecraft consisted of a combined "
    "command and service module (CSM) and the Apollo Lunar Module (LM)."
)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def load_summarizer(model_name: str = DEFAULT_MODEL):
    """Load the summarization pipeline pinned to CPU."""
    return pipeline(
        task="summarization",
        model=model_name,
        device=-1,  # -1 == CPU in transformers.pipeline
    )


def summarize(text: str, model_name: str = DEFAULT_MODEL) -> str:
    """Summarize a single text input and return the summary string."""
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    # Truncate to avoid blowing past the model's token budget on CPU.
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    summarizer = load_summarizer(model_name)
    result = summarizer(
        text,
        max_length=130,
        min_length=30,
        do_sample=False,
    )
    return result[0]["summary_text"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only text summarizer using Hugging Face Transformers.",
    )
    parser.add_argument(
        "-t", "--text",
        help="Text to summarize. If omitted, reads stdin or uses sample.",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"HF model name (default: {DEFAULT_MODEL})",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    # Resolve input source: --text > stdin > sample.
    if args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        text = SAMPLE_TEXT

    try:
        if torch.cuda.is_available():
            print("Warning: CUDA detected, but CPU mode is enforced.",
                  file=sys.stderr)
        summary = summarize(text, model_name=args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
