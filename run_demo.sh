#!/bin/bash

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    venv/bin/pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
fi

# Add to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Default port
DEFAULT_PORT=${GRADIO_PORT:-7860}

# Parse arguments
PYTHON_ARGS=()
HAS_PORT=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --port|--server_port)
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

if [ $HAS_PORT -eq 0 ]; then
    PYTHON_ARGS+=("--server_port" "$DEFAULT_PORT")
fi

echo "Starting Genres in Genres on port ${PYTHON_ARGS[1]}..."
venv/bin/python3 app.py "${PYTHON_ARGS[@]}"
