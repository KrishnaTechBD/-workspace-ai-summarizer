"""Tests for summarizer.py — CPU mode smoke tests.

These tests avoid downloading any Hugging Face model by mocking the
``pipeline`` factory, so the test suite stays fast and offline-friendly.
"""
import os
import sys
from unittest import mock

import pytest

# Make the project root importable so `import summarizer` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import summarizer  # noqa: E402


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------
def test_sample_text_is_non_empty_string():
    """The built-in CI sample text must be a non-empty string."""
    assert isinstance(summarizer.SAMPLE_TEXT, str)
    assert len(summarizer.SAMPLE_TEXT) > 100


def test_default_model_is_string():
    """Default model name must be a non-empty string."""
    assert isinstance(summarizer.DEFAULT_MODEL, str)
    assert summarizer.DEFAULT_MODEL


def test_max_input_chars_is_positive():
    """The char-truncation threshold must be a positive int."""
    assert isinstance(summarizer.MAX_INPUT_CHARS, int)
    assert summarizer.MAX_INPUT_CHARS > 0


# ---------------------------------------------------------------------------
# CPU enforcement
# ---------------------------------------------------------------------------
def test_pipeline_is_loaded_with_cpu_device():
    """``pipeline`` must be called with device=-1 (CPU)."""
    with mock.patch("summarizer.pipeline") as mock_pipeline:
        summarizer.load_summarizer()
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs.get("device") == -1
        assert kwargs.get("task") == "summarization"


def test_cuda_visible_devices_is_forced_empty():
    """summarizer.py must force CUDA_VISIBLE_DEVICES='' at import time."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


# ---------------------------------------------------------------------------
# summarize() behavior
# ---------------------------------------------------------------------------
def test_summarize_rejects_empty_string():
    """Empty input should raise ValueError."""
    with pytest.raises(ValueError):
        summarizer.summarize("")


def test_summarize_rejects_whitespace_only():
    """Whitespace-only input should raise ValueError."""
    with pytest.raises(ValueError):
        summarizer.summarize("   \n\t  ")


def test_summarize_truncates_long_input():
    """Inputs longer than MAX_INPUT_CHARS should be truncated before inference."""
    long_text = "lorem ipsum " * (summarizer.MAX_INPUT_CHARS // 12 + 50)

    fake_summarizer = mock.MagicMock(return_value=[{"summary_text": "ok"}])
    with mock.patch("summarizer.load_summarizer", return_value=fake_summarizer):
        result = summarizer.summarize(long_text)

    passed_text = fake_summarizer.call_args.args[0]
    assert len(passed_text) <= summarizer.MAX_INPUT_CHARS
    assert result == "ok"


def test_summarize_returns_pipeline_output():
    """summarize() should return the model's summary text."""
    fake_summarizer = mock.MagicMock(
        return_value=[{"summary_text": "a short summary"}]
    )
    with mock.patch("summarizer.load_summarizer", return_value=fake_summarizer):
        out = summarizer.summarize("some input text " * 20)

    assert out == "a short summary"
    fake_summarizer.assert_called_once()
