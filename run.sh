#!/bin/bash
# AutoKatha Unified Launcher
# Starts ComfyUI + Ollama + AutoKatha web interface with a single command

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_APP="/Applications/ComfyUI.app"
COMFYUI_PORT=8000
AUTOKATHA_PORT=7861
OLLAMA_URL="http://localhost:11434"

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🎬 AutoKatha Launcher              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if a port is in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=0
    
    echo -n "   Waiting for $name..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -n "."
    done
    echo -e " ${RED}✗ (timeout)${NC}"
    return 1
}

# Step 1: Check/Start Ollama
echo -e "${YELLOW}[1/4]${NC} Checking Ollama..."
if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓${NC} Ollama is already running"
else
    echo "   Starting Ollama..."
    if command -v ollama &> /dev/null; then
        ollama serve > /dev/null 2>&1 &
        sleep 2
        if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
            echo -e "   ${GREEN}✓${NC} Ollama started"
        else
            echo -e "   ${RED}✗${NC} Failed to start Ollama"
            echo "   Please start Ollama manually: ollama serve"
        fi
    else
        echo -e "   ${YELLOW}⚠${NC} Ollama not found. Scene splitting will use fallback."
    fi
fi

# Step 2: Start ComfyUI
echo -e "${YELLOW}[2/4]${NC} Checking ComfyUI..."
if check_port $COMFYUI_PORT; then
    echo -e "   ${GREEN}✓${NC} ComfyUI is already running on port $COMFYUI_PORT"
else
    if [ -d "$COMFYUI_APP" ]; then
        echo "   Launching ComfyUI.app..."
        open "$COMFYUI_APP"
        wait_for_service "http://127.0.0.1:$COMFYUI_PORT/system_stats" "ComfyUI" 60
    else
        echo -e "   ${YELLOW}⚠${NC} ComfyUI.app not found at $COMFYUI_APP"
        echo "   Image generation will use diffusers fallback"
    fi
fi

# Step 3: Activate virtual environment and start AutoKatha
echo -e "${YELLOW}[3/4]${NC} Starting AutoKatha..."

# Kill any existing instance on the port
if check_port $AUTOKATHA_PORT; then
    echo "   Stopping existing instance on port $AUTOKATHA_PORT..."
    lsof -ti :$AUTOKATHA_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Activate venv and run
cd "$SCRIPT_DIR"
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo -e "   ${RED}✗${NC} Virtual environment not found!"
    echo "   Please run: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Start AutoKatha in background
python app_text.py > /dev/null 2>&1 &
AUTOKATHA_PID=$!

# Wait for it to be ready
wait_for_service "http://127.0.0.1:$AUTOKATHA_PORT" "AutoKatha" 30

# Step 4: Open browser
echo -e "${YELLOW}[4/4]${NC} Opening browser..."
sleep 1
open "http://127.0.0.1:$AUTOKATHA_PORT"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         🎬 AutoKatha is Ready!             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "   📍 Web UI:    ${BLUE}http://127.0.0.1:$AUTOKATHA_PORT${NC}"
echo -e "   🎨 ComfyUI:   ${BLUE}http://127.0.0.1:$COMFYUI_PORT${NC}"
echo -e "   🤖 Ollama:    ${BLUE}$OLLAMA_URL${NC}"
echo ""
echo -e "   Press ${YELLOW}Ctrl+C${NC} to stop AutoKatha"
echo ""

# Keep script running and forward signals
trap "echo ''; echo 'Shutting down...'; kill $AUTOKATHA_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for AutoKatha process
wait $AUTOKATHA_PID
