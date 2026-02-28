# Linux Setup Notes

## First-time Linux Setup

This project was successfully set up on Linux for the first time. Here are the notes:

### Environment
- Platform: Linux
- Python version: 3.10.12
- Ollama version: 0.16.3
- Virtual environment: .venv (created by configure.sh)

### Setup Process
1. Made configure.sh executable: `chmod +x configure.sh`
2. Ran configure script: `./configure.sh`
   - Script detected Python 3.10.12 automatically
   - Created virtual environment at .venv
   - Installed dependencies from requirements.txt
3. Activated virtual environment: `source .venv/bin/activate`
4. Tested functionality:
   - `python -m ollama_mgr.cli check` - Confirmed Ollama is running
   - `python -m ollama_mgr.cli list` - Successfully listed 12 models
   - `python run.py --help` - Confirmed entry point works

### Models Available
The system has access to 12 models including:
- glm-5:cloud
- kimi-k2.5:cloud  
- minimax-m2.5:cloud
- llama3.1:8b
- qwen2.5-coder:7b-instruct
- deepseek-coder:latest
- deepseek-r1:7b
- llava:latest
- llava:7b
- llama3:latest
- phi3:latest
- mistral:latest

### Notes
- No Linux-specific modifications were needed
- The configure.sh script worked perfectly on Linux
- All functionality tested successfully
- No test suite found in the project (no test*.py files)

### Usage
```bash
# Activate environment
source .venv/bin/activate

# Check Ollama status
python -m ollama_mgr.cli check

# List models
python -m ollama_mgr.cli list

# Interactive benchmark
python -m ollama_mgr.cli run

# Batch benchmark
python -m ollama_mgr.cli benchmark

# Or use run.py entry point
python run.py check
python run.py list
python run.py benchmark
```