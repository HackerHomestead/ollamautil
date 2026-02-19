# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **Turn-based chat (`chat`)** – Two models debate each other in a split-view UI with live streaming output. Select Player 1 (left) and Player 2 (right) models interactively or via CLI flags. Each model's response feeds into the other's next turn.
  - Split-view panel showing both models' responses side-by-side
  - `--player1` / `-1` and `--player2` / `-2` for model selection
  - `--prompt` for initial prompt (or prompted if omitted)
  - `--exchanges` / `-e` for number of turns per model (default: 5)
  - `--timeout` / `-t` for timeout in seconds (default: 3600 = 1 hour)
  - Controls: **p** = pause/resume, **s** = skip next turn, **q** = quit
  - Chat logs saved to `/tmp` with timestamps relative to session start
  - After session: option to summarize with another model (select model + custom prompt like "Who won this debate?")
- **Chat API support** – Added `chat()` method to `OllamaClient` for chat completion. The `generate()` method now accepts `use_chat_api=True` to route prompts via `/api/chat` instead of `/api/generate`. Benchmark automatically detects models that need chat format (e.g., DeepSeek-R1) based on model name prefixes.
- **Interactive run (`run`)**
  - Checkbox model selection; run one model at a time with live streaming output
  - Live output: model response streams in real time and is rendered as Markdown (code blocks, **bold**, lists, etc.)
  - Pause between models with countdown (default 15s; `--pause` / `-p`, use `0` to disable)
  - During run: **n** = skip to next model (marked “cut short by user” in results; elapsed time, estimated token count, and speed still recorded), **p** = pause/resume display
  - When a run is cut short by user: benchmark is preserved (elapsed time; token count estimated from captured text if API didn’t send final count; tokens/sec from those); token count shown as “(est.)” in table and report
  - During delay: **n** = skip delay, **p** = pause/resume countdown
  - After run: option to delete failed models (checkbox; choose which to remove)
  - “Run again?” prompt to loop the interactive flow
  - Overall run time shown in summary
  - Environment (CPU, memory, GPU, platform) captured and printed after results; included in report when using `-o`
  - Results table sorted by status (passed first), then speed, then tokens
  - Report file (`-o`): env, prompt, per-model stats; model output written as Markdown so viewers render formatting
- **Benchmark (`benchmark`)**
  - Batch run with env and nvidia-smi sampling; report to stdout or file
  - nvidia-smi only run when present on the system (no subprocess if not on PATH)
- **Prune (`prune`)** – delete models by name (positional or `--models` list)
- **Check** and **list** – health check and list installed models

### Notes

- **Future:** Support for vision/image models (images or files as input) is planned; see TODOs in code.
- **Future:** Use SQLite to retain all results for later recall and display.
- **Future:** Maintain Markdown rendering when output scrolls off screen (currently the last-N-lines window is shown as plain text to avoid broken formatting; goal is to truncate at safe boundaries or use a scrollable view so rendering stays correct).
