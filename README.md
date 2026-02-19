# Ollama Model Manager

A Python tool that interacts with the [Ollama](https://ollama.com) API to:

- **Check** that Ollama is running
- **List** all installed models
- **Benchmark** every model (test run + quick performance measure)
- **Report** what worked and what didn’t, plus environment (CPU, memory, GPU) and `nvidia-smi` usage sampled every 10 seconds
- **Prune** (delete) models that don’t work or you no longer want

Primary use case: automatically test downloaded Ollama models and remove ones that fail or are too slow.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running (e.g. `ollama serve`)
- Optional: NVIDIA GPU and `nvidia-smi` for GPU/env stats in the report

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

### Interactive run (select models, then benchmark one at a time)

Pick models with a **checkbox menu**, then run each with a test prompt. Verbose live output so you can see progress when models are slow.

```bash
python -m ollama_mgr.cli run
```

- Default prompt: `write hello world in lisp`
- Custom prompt: `python -m ollama_mgr.cli run --prompt "Your prompt"`
- Non-interactive (no checkboxes): `python -m ollama_mgr.cli run --models "model1,model2"`
- Write results to a markdown file (prompt + each model’s stats and output, for comparison): `python -m ollama_mgr.cli run -o report.md`
- Suppress live UI (minimal progress only): `python -m ollama_mgr.cli run --quiet` or `-q`

### Benchmark all models and show report

Runs each model with a short test prompt, records timing and tokens/sec, captures CPU/memory/GPU config and runs `nvidia-smi` at least every 10 seconds during the run. Report is printed to stdout.

```bash
python -m ollama_mgr.cli benchmark
```

Save the report to a file:

```bash
python -m ollama_mgr.cli benchmark --report report.md
```

Options:

- `--prompt "Your test prompt"` – prompt used for each model (default: `Say exactly: OK`)
- `--nvidia-smi-interval 10` – run `nvidia-smi` every N seconds (default: 10)
- `--report FILE` / `-o FILE` – write report to `FILE`

### Prune (delete) models

By name (positional or comma-separated):

```bash
python -m ollama_mgr.cli prune llama2:7b codellama
python -m ollama_mgr.cli prune --models "llama2:7b,codellama"
```

## Report contents

The benchmark report includes:

1. **Environment** – CPU (cores, frequency, brand), RAM (total/available/percent), GPU(s) (name, memory, driver), platform
2. **nvidia-smi usage during run** – one line per sample (every 10s) plus raw output
3. **Model results** – passed (with elapsed time and tokens/sec) and failed (with error message)

Use this to decide which models to prune (e.g. failed or too slow), then run `prune` with those names.
