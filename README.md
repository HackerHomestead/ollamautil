# Ollama Model Manager

A Python tool that interacts with the [Ollama](https://ollama.com) API to:

- **Check** that Ollama is running
- **List** all installed models
- **Run** an interactive benchmark: pick models (checkboxes), run one at a time with live streaming output, optional pause between models, then optionally delete failed models or run again
- **Benchmark** every model (batch test + performance measure) with environment and `nvidia-smi` sampling
- **Prune** (delete) models by name

Primary use case: test downloaded Ollama models, compare outputs and speed, and remove ones that fail or you no longer want.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running (e.g. `ollama serve`)
- Optional: NVIDIA GPU and `nvidia-smi` on PATH for GPU/env stats (only used if present)

## Install

**Recommended: use the configure script (creates a virtual environment and installs deps):**

```bash
cd /path/to/ollama-model-mgr
./configure.sh
source .venv/bin/activate
```

Options: `./configure.sh --force` to recreate the venv; `PYTHON=python3.10 ./configure.sh` if auto-detect fails.

**Manual install:**

```bash
cd /path/to/ollama-model-mgr
pip install -r requirements.txt
```

## Usage

All commands default to `http://localhost:11434`. Override with `--base-url` if needed.

### Check if Ollama is working

```bash
python -m ollama_mgr.cli check
```

### List all models

```bash
python -m ollama_mgr.cli list
```

### Interactive run (recommended)

Pick models with a **checkbox menu**, then run each with a test prompt. You see **live streaming output** from each model (with markdown rendered: code blocks, bold, lists, etc.), overall run time, and environment (CPU, memory, GPU). After the run you can optionally delete failed models (checkbox) and choose to run again.

```bash
python -m ollama_mgr.cli run
```

**Options:**

- `--prompt "Your prompt"` – test prompt (default: you’re prompted; or `write hello world in lisp` when selecting models)
- `--models "model1,model2"` / `-m` – run these models (no checkbox)
- `--report FILE` / `-o FILE` – write a markdown report (env + prompt + each model’s stats and output; output is written as markdown so viewers render it)
- `--pause SECONDS` / `-p SECONDS` – pause between models with countdown (default: 15). Use `0` to disable.
- `--quiet` / `-q` – minimal progress (no live UI)

**During a model run:**

- **n** – skip to next model (run is marked “cut short by user” in results; elapsed time, estimated token count, and speed are still recorded)
- **p** – pause the display (freeze output so you can read); press **p** again to resume

**During the delay between models:**

- **n** – skip the delay and start the next model
- **p** – pause the countdown; **p** again to resume

The results table is sorted by status (passed first), then speed, then tokens. When you skip a model with **n**, its row shows “(cut short by user)” and token count is shown as estimated “(est.)” if the API didn’t send a final count. At the end you can select which failed models to delete (checkboxes) and choose whether to run again.

### Benchmark all models and show report

Runs each model with a short test prompt, records timing and tokens/sec, captures CPU/memory/GPU and runs `nvidia-smi` every N seconds during the run (only if `nvidia-smi` is on PATH). Report is printed to stdout.

```bash
python -m ollama_mgr.cli benchmark
```

Save the report to a file:

```bash
python -m ollama_mgr.cli benchmark --report report.md
```

Options:

- `--prompt "Your test prompt"` – prompt for each model (default: `Say exactly: OK`)
- `--nvidia-smi-interval N` – sample `nvidia-smi` every N seconds (default: 10)
- `--report FILE` / `-o FILE` – write report to `FILE`

### Prune (delete) models

By name (positional or comma-separated):

```bash
python -m ollama_mgr.cli prune llama2:7b codellama
python -m ollama_mgr.cli prune --models "llama2:7b,codellama"
```

## Report contents

**Interactive run report** (with `-o`):

- Environment (CPU, memory, GPU(s), platform)
- Prompt
- Per model: status, time, tokens, speed; output as markdown (so code blocks and formatting render in viewers). If a run was cut short by user, that is noted.

**Benchmark report** (batch):

- Environment – CPU (cores, frequency, brand), RAM (total/available/percent), GPU(s) (name, memory, driver), platform
- nvidia-smi usage during run – one line per sample (only if `nvidia-smi` is available)
- Model results – passed (elapsed time, tokens/sec) and failed (error message)

Use this to decide which models to prune, then run `prune` with those names.

## Future / planned

- **Vision/image models** – Support for images or files as input (see TODOs in code).
- **SQLite storage** – Retain all run results in SQLite for later recall and display.
- **Turn-based chat** – Chat between models: choose one or more models, give each an initial prompt, then have them send responses to each other in turns.
- **Markdown when scrolling** – Keep full Markdown rendering when output scrolls off screen (e.g. truncate at safe block boundaries or use a scrollable view instead of plain-text fallback for the last N lines).
