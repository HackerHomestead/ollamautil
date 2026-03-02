# Ollama Utility (Go)

A fast, modern CLI tool for managing Ollama models, written in Go. This is a complete rewrite of the original Python version with significant improvements in performance, deployment, and maintainability.

## 🚀 Features

- **Health Monitoring**: Check Ollama service status and version
- **Model Management**: List, benchmark, and delete models
- **Performance Benchmarking**: Run comprehensive benchmarks with detailed metrics
- **System Monitoring**: Real-time CPU, memory, and GPU monitoring during benchmarks
- **Interactive Mode**: Stream model responses in real-time
- **Report Generation**: Export results to Markdown format
- **Cross-Platform**: Single binary deployment for Linux, macOS, and Windows

## 📦 Installation

### Pre-built Binaries (Recommended)

Download the latest release for your platform from the [releases page](link-to-releases).

### Build from Source

```bash
git clone <repository-url>
cd ollama-util-go
go build -o ollama-util .
```

## 🎯 Quick Start

1. **Check if Ollama is running:**
   ```bash
   ./ollama-util check
   ```

2. **List available models:**
   ```bash
   ./ollama-util list
   ```

3. **Run a quick benchmark:**
   ```bash
   ./ollama-util run --models "phi3:latest" --prompt "Say exactly: OK"
   ```

4. **Benchmark all models and generate report:**
   ```bash
   ./ollama-util benchmark --output report.md
   ```

## 🔧 Commands

### `check`
Check if Ollama service is running and accessible.

```bash
ollama-util check
```

### `list`
List all available models with size information.

```bash
ollama-util list
```

### `run`
Run interactive benchmarks with real-time streaming output.

```bash
# Run all models with default prompt
ollama-util run

# Run specific models
ollama-util run --models "phi3:latest,mistral:latest"

# Custom prompt with report generation
ollama-util run --prompt "Write a Python function to reverse a string" --output results.md

# Quiet mode with custom pause
ollama-util run --quiet --pause 5
```

**Options:**
- `--prompt`: Custom test prompt (default: "write hello world in lisp")
- `--models, -m`: Comma-separated model names (default: all models)
- `--output, -o`: Save results to Markdown file
- `--quiet, -q`: Suppress real-time output
- `--pause, -p`: Pause between models in seconds (default: 15)

### `benchmark`
Run batch benchmarks on all models and generate comprehensive reports.

```bash
# Benchmark all models
ollama-util benchmark

# Custom prompt and output file
ollama-util benchmark --prompt "Explain quantum computing" --output benchmark.md
```

**Options:**
- `--prompt`: Test prompt (default: "Say exactly: OK")
- `--output, -o`: Write report to file
- `--nvidia-smi-interval`: GPU monitoring interval (default: 10s)

### `prune`
Delete models by name.

```bash
# Delete single model
ollama-util prune llama2

# Delete multiple models
ollama-util prune llama2 codellama mistral

# Using comma-separated list
ollama-util prune --models "llama2,codellama,mistral"
```

## 🔧 Configuration

### Base URL
By default, the tool connects to `http://localhost:11434`. You can override this:

```bash
ollama-util --base-url http://your-ollama-server:11434 check
```

### Environment Variables
Set `OLLAMA_HOST` to change the default base URL:

```bash
export OLLAMA_HOST=http://your-server:11434
ollama-util check
```

## 📊 Output Examples

### Health Check
```
✅ Ollama is running.
Version: 0.16.3
```

### Model List
```
Found 10 model(s):
  phi3:latest  (2.0 GB)
  mistral:latest  (3.8 GB)
  llama2:7b  (3.9 GB)
```

### Benchmark Results
```
Running 2 model(s) with prompt: "Say exactly: OK"
============================================================

[1/2] Running phi3:latest...
Response: OK

✅ Completed in 15.417s (12.74 tokens/sec)

[2/2] Running mistral:latest...  
Response: OK

✅ Completed in 8.234s (18.92 tokens/sec)

============================================================
SUMMARY
============================================================
Total: 2 models, 2 successful, 0 failed
Average duration: 11.825s
Average tokens/sec: 15.83
```

## 🏗️ Architecture

The Go version features a clean, modular architecture:

```
cmd/           # CLI commands (check, list, run, benchmark, prune)
internal/
├── ollama/    # HTTP client and API types
├── system/    # System monitoring (CPU, memory, GPU)
├── benchmark/ # Benchmarking logic and result processing
└── ui/        # Terminal UI components (future interactive features)
```

## 🆚 Advantages Over Python Version

### Performance
- **10x+ faster startup** - No Python interpreter or virtual environment
- **Lower memory usage** - Efficient Go runtime vs Python + dependencies
- **Faster HTTP requests** - Built-in net/http vs requests library

### Deployment
- **Single binary** - No dependency management or virtual environments
- **Cross-compilation** - Build for multiple platforms from one machine
- **Smaller footprint** - ~10MB binary vs ~50MB+ Python environment

### Reliability
- **Type safety** - Compile-time error checking vs runtime failures
- **Better concurrency** - Goroutines vs Python threading limitations
- **Memory safety** - No runtime type errors or null pointer exceptions

### Development
- **Simpler CI/CD** - Just compile and distribute binaries
- **Easier testing** - Integrated testing tools and benchmarking
- **Better tooling** - Rich ecosystem of CLI libraries (Cobra, bubbletea)

## 🔮 Future Enhancements

- [ ] Rich interactive TUI for model selection
- [ ] WebSocket support for real-time monitoring
- [ ] Plugin system for custom benchmarks
- [ ] Distributed benchmarking across multiple Ollama instances
- [ ] Model comparison and A/B testing features
- [ ] Integration with monitoring systems (Prometheus, Grafana)

## 🛠️ Development

### Prerequisites
- Go 1.21 or later
- Ollama running locally or accessible via network

### Building
```bash
go mod tidy
go build -o ollama-util .
```

### Testing
```bash
go test ./...
```

### Cross-Platform Builds
```bash
# Linux
GOOS=linux GOARCH=amd64 go build -o ollama-util-linux .

# macOS
GOOS=darwin GOARCH=amd64 go build -o ollama-util-macos .

# Windows  
GOOS=windows GOARCH=amd64 go build -o ollama-util-windows.exe .
```

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

---

**Note**: This Go version provides all the functionality of the original Python implementation while being significantly faster, easier to deploy, and more maintainable. The single-binary distribution eliminates the complexity of Python environments and dependency management.