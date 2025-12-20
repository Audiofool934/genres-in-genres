#!/bin/bash

# Set CUDA_VISIBLE_DEVICES FIRST, before any Python imports
# This must be set before torch is imported
export CUDA_VISIBLE_DEVICES=0

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Change to subproject directory
cd "$SCRIPT_DIR"

# Ensure venv in subproject root
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
    "$PROJECT_ROOT/venv/bin/pip" install --upgrade pip
    "$PROJECT_ROOT/venv/bin/pip" install -r "$PROJECT_ROOT/requirements.txt"
fi

echo "Starting Genres in Genres App..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Add current directory to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default port (can be overridden by --port or GRADIO_PORT env var)
# Usage: GRADIO_PORT=8080 ./run_demo.sh
# Or: ./run_demo.sh --port 8080
DEFAULT_PORT=${GRADIO_PORT:-7860}

# Parse command line arguments
# Supports both --port (converts to --server_port) and --server_port (passes through)
PYTHON_ARGS=()
HAS_PORT=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PYTHON_ARGS+=("--server_port" "$2")
            HAS_PORT=1
            shift 2
            ;;
        --server_port)
            PYTHON_ARGS+=("--server_port" "$2")
            HAS_PORT=1
            shift 2
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# If no port specified, use default
if [ $HAS_PORT -eq 0 ]; then
    PYTHON_ARGS+=("--server_port" "$DEFAULT_PORT")
fi

# Launch the app
"$PROJECT_ROOT/venv/bin/python3" app.py "${PYTHON_ARGS[@]}"
