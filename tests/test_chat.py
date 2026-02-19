"""Tests for turn-based chat module."""

import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
import time

from ollama_mgr.chat import (
    _format_timestamp,
    _setup_log_file,
    _init_log_file,
    _append_to_log,
    _render_chat_panel,
    run_turn_chat,
    run_summary,
)


class TestFormatTimestamp:
    def test_zero_seconds(self):
        assert _format_timestamp(0) == "00:00"

    def test_seconds_under_minute(self):
        assert _format_timestamp(45) == "00:45"

    def test_one_minute(self):
        assert _format_timestamp(60) == "01:00"

    def test_minutes_and_seconds(self):
        assert _format_timestamp(125) == "02:05"

    def test_large_value(self):
        assert _format_timestamp(3600) == "60:00"


class TestLogFile:
    def test_setup_log_file(self):
        path, path_str = _setup_log_file()
        assert path.name.startswith("chat_")
        assert path.name.endswith(".md")
        assert str(path.parent) == "/tmp"
        assert path_str == str(path)

    def test_init_log_file(self, tmp_path):
        log_file = tmp_path / "test_chat.md"
        _init_log_file(log_file, "model1", "model2", "test prompt", 0.0)
        content = log_file.read_text()
        assert "# Turn-Based Chat Session" in content
        assert "model1" in content
        assert "model2" in content
        assert "test prompt" in content

    def test_append_to_log(self, tmp_path):
        log_file = tmp_path / "test_chat.md"
        _init_log_file(log_file, "model1", "model2", "test prompt", 0.0)
        _append_to_log(log_file, 1, "model1", "Hello world", 10.0)
        content = log_file.read_text()
        assert "### Exchange 1 — model1" in content
        assert "Hello world" in content
        assert "00:10" in content


class TestRenderChatPanel:
    def test_empty_content(self):
        panel = _render_chat_panel("Test", "")
        assert "Test" in str(panel.title)

    def test_short_content(self):
        panel = _render_chat_panel("Test", "Hello")
        assert "Test" in str(panel.title)

    def test_long_content_truncates(self):
        long_content = "\n".join([f"line {i}" for i in range(50)])
        panel = _render_chat_panel("Test", long_content, max_lines=30)
        assert "Test" in str(panel.title)


class TestRunTurnChat:
    @patch("ollama_mgr.chat._init_log_file")
    @patch("ollama_mgr.chat._append_to_log")
    @patch("ollama_mgr.chat._setup_log_file")
    def test_basic_conversation(self, mock_log_setup, mock_append, mock_init):
        mock_log_setup.return_value = (tempfile.mktemp(suffix=".md"), "/tmp/test.md")

        from ollama_mgr.chat import run_turn_chat

        client = Mock()

        def mock_chat(model, messages, stream=False, options=None):
            exchange_num = len([m for m in messages if m.get("role") == "assistant"])
            return (True, None, None, iter([{"response": f"Response {exchange_num + 1}"}]))

        client.chat.side_effect = mock_chat

        log, elapsed, was_quit = run_turn_chat(
            client,
            "model1",
            "model2",
            "initial prompt",
            max_exchanges=2,
            live=False,
        )

        assert len(log) == 4
        assert was_quit is False

    @patch("ollama_mgr.chat._init_log_file")
    @patch("ollama_mgr.chat._append_to_log")
    @patch("ollama_mgr.chat._setup_log_file")
    def test_handles_stream_error(self, mock_log_setup, mock_append, mock_init):
        mock_log_setup.return_value = (tempfile.mktemp(suffix=".md"), "/tmp/test.md")

        from ollama_mgr.chat import run_turn_chat

        client = Mock()

        def mock_chat(model, messages, stream=False, options=None):
            return (False, "error message", None, None)

        client.chat.side_effect = mock_chat

        log, elapsed, was_quit = run_turn_chat(
            client,
            "model1",
            "model2",
            "initial prompt",
            max_exchanges=1,
            live=False,
        )

        assert len(log) == 0


class TestRunSummary:
    @patch("ollama_mgr.chat._stream_chat_turn")
    def test_summary_with_context(self, mock_stream):
        mock_stream.return_value = (True, None, "Model 1 made better arguments.")

        client = Mock()
        conversation_log = [
            {"player": "model1", "content": "I think AI will help humanity.", "elapsed": 10.0},
            {"player": "model2", "content": "I disagree, AI poses risks.", "elapsed": 20.0},
        ]

        ok, err, summary = run_summary(
            client,
            "model3",
            conversation_log,
            "Who won?",
        )

        assert ok is True
        assert summary == "Model 1 made better arguments."
        mock_stream.assert_called_once()
        call_args = mock_stream.call_args
        messages = call_args[0][2]
        assert "model1" in messages[0]["content"]
        assert "model2" in messages[0]["content"]

    @patch("ollama_mgr.chat._stream_chat_turn")
    def test_summary_handles_error(self, mock_stream):
        mock_stream.return_value = (False, "API error", "")

        client = Mock()
        conversation_log = [{"player": "model1", "content": "Hello", "elapsed": 10.0}]

        ok, err, summary = run_summary(client, "model2", conversation_log, "Summarize")

        assert ok is False
        assert err == "API error"


class TestKeyReaderThread:
    def test_key_reader_initialization(self):
        from ollama_mgr.chat import _chat_key_reader
        holder = {"active": True, "quit": False, "paused": False}
        assert holder["active"] is True
        assert holder["quit"] is False
