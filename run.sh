#!/bin/bash

# Martini — Blind Source Separation Runner
# This script automates environment setup and application execution.

# --- Configuration ---
VENV_PATH=".venv"
PYTHON_BIN="$VENV_PATH/bin/python"

# --- Colors for Output ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function show_help() {
    echo -e "${BLUE}Martini Management Script${NC}"
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  gui       Launch the PySide6 Graphical User Interface (Default)"
    echo "  setup     Install dependencies and generate synthetic test data"
    echo "  info      Show information about found stems in data/raw/"
    echo "  separate  Run a quick time-domain separation via CLI (2s duration)"
    echo "  help      Show this help message"
}

function check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Virtual environment not found at $VENV_PATH.${NC}"
        echo "Please run './run.sh setup' first."
        exit 1
    fi
}

case "$1" in
    setup)
        echo -e "${YELLOW}Setting up Martini...${NC}"
        if [ ! -d "$VENV_PATH" ]; then
            echo "Creating virtual environment..."
            uv venv
        fi
        echo "Installing dependencies..."
        uv sync
        echo "Generating synthetic test data..."
        $PYTHON_BIN downloads/fetch_tracks.py --synthetic
        echo -e "${GREEN}Setup complete!${NC}"
        ;;

    gui | "")
        check_venv
        echo -e "${GREEN}Launching Martini GUI...${NC}"
        $PYTHON_BIN gui.py
        ;;

    info)
        check_venv
        $PYTHON_BIN main.py info
        ;;

    separate)
        check_venv
        echo -e "${YELLOW}Running CLI separation (Time Domain, 2s)...${NC}"
        $PYTHON_BIN main.py separate --duration 2.0 --mode time
        ;;

    help)
        show_help
        ;;

    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
