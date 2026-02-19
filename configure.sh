#!/usr/bin/env bash
# Autoconfigure script for Ollama Model Manager
# Creates a Python virtual environment and installs dependencies.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
VENV_DIR="${VENV_DIR:-.venv}"
MIN_PYTHON_VERSION=3.9

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Configure the Ollama Model Manager environment (Python virtualenv + deps).

Options:
  -v, --venv DIR    Use DIR as the virtualenv (default: .venv)
  -f, --force       Recreate venv if it already exists
  -h, --help        Show this help

Environment:
  PYTHON            Use this Python (e.g. python3.10) if auto-detect fails
  VENV_DIR          Default venv path (overridden by -v/--venv)
EOF
}

# Parse options
FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--venv) VENV_DIR="$2"; shift 2 ;;
        -f|--force) FORCE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Find Python 3.9+ that can create a working venv (with pip)
find_python() {
    local try_venv
    try_venv=$(mktemp -d 2>/dev/null || echo "/tmp/om_mgr_venv_$$")
    for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$cmd" &>/dev/null; then
            local version
            version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || continue
            if awk -v min="$MIN_PYTHON_VERSION" -v v="$version" 'BEGIN { exit (v+0 >= min+0) ? 0 : 1 }' 2>/dev/null; then
                rm -rf "$try_venv"
                if "$cmd" -m venv "$try_venv" 2>/dev/null && "$try_venv/bin/python" -m pip --version &>/dev/null; then
                    rm -rf "$try_venv"
                    echo "$cmd"
                    return
                fi
                rm -rf "$try_venv"
            fi
        fi
    done
    rm -rf "$try_venv" 2>/dev/null || true
    return 1
}

if [[ -n "${PYTHON:-}" ]]; then
    if ! "$PYTHON" -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
        echo "Error: PYTHON=$PYTHON is not Python ${MIN_PYTHON_VERSION}+." >&2
        exit 1
    fi
else
    PYTHON=$(find_python) || {
        echo "Error: No Python ${MIN_PYTHON_VERSION}+ with working venv+pip found." >&2
        echo "Try: export PYTHON=python3.10 && $0" >&2
        echo "Or install: python3.9-venv / python3-venv" >&2
        exit 1
    }
fi

echo "Using: $PYTHON ($($PYTHON --version 2>&1))"
echo "Virtual environment: $VENV_DIR"

if [[ -d "$VENV_DIR" ]]; then
    if [[ "$FORCE" == true ]]; then
        echo "Removing existing venv (--force)."
        rm -rf "$VENV_DIR"
    else
        echo "Virtual environment already exists at $VENV_DIR"
        echo "Activate with: source $VENV_DIR/bin/activate"
        echo "To recreate, run: $0 --force"
        echo ""
        echo "Quick test:"
        echo "  source $VENV_DIR/bin/activate && python -m ollama_mgr.cli check"
        exit 0
    fi
fi

echo "Creating virtual environment..."
if "$PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
    :
elif "$PYTHON" -m venv "$VENV_DIR" --without-pip 2>/dev/null; then
    echo "Venv created without pip (ensurepip missing). Trying virtualenv..."
    rm -rf "$VENV_DIR"
    if "$PYTHON" -m pip install --user virtualenv 2>/dev/null || "$PYTHON" -m pip install virtualenv 2>/dev/null; then
        "$PYTHON" -m virtualenv "$VENV_DIR"
    else
        echo "Error: Could not create venv (ensurepip missing) and 'virtualenv' not available." >&2
        echo "Install one of: python3.9-venv, python3-venv, or: $PYTHON -m pip install virtualenv" >&2
        exit 1
    fi
else
    echo "Error: Failed to create virtual environment." >&2
    exit 1
fi

# Activate and install (bash/zsh)
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install -q --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Configured successfully."
echo ""
echo "Activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run:"
echo "  python -m ollama_mgr.cli check   # verify Ollama is running"
echo "  python -m ollama_mgr.cli list     # list models"
echo "  python -m ollama_mgr.cli benchmark   # benchmark all models"
echo "  python run.py check              # or use the run.py entry point"
echo ""

# Quick sanity check
if python -c "import ollama_mgr; print('ollama_mgr version:', ollama_mgr.__version__)" 2>/dev/null; then
    echo "Sanity check passed."
else
    echo "Warning: import check failed. Activate the venv and run: python -m ollama_mgr.cli check"
fi
