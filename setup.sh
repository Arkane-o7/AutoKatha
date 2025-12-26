#!/bin/bash
# AutoKatha Setup Script for macOS

set -e

echo "========================================"
echo "  AutoKatha Setup Script"
echo "  Comic to Video Pipeline"
echo "========================================"
echo ""

# Check for Homebrew
if ! command -v /opt/homebrew/bin/brew &> /dev/null; then
    echo "❌ Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew found"
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
/opt/homebrew/bin/brew install ffmpeg poppler || true

# Find best Python version (prefer 3.11, 3.12, 3.13, then fall back to python3)
PYTHON_CMD=""
for py in python3.13 python3.12 python3.11; do
    if [ -x "/opt/homebrew/bin/$py" ]; then
        PYTHON_CMD="/opt/homebrew/bin/$py"
        break
    elif command -v $py &> /dev/null; then
        PYTHON_CMD=$py
        break
    fi
done

# Fall back to python3 if no specific version found
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
    echo "✅ Python $PYTHON_VERSION found ($PYTHON_CMD)"
else
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    echo "   Install with: brew install python@3.11"
    exit 1
fi

# Create virtual environment
echo ""
echo "🐍 Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support
echo ""
echo "🔥 Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# Install main dependencies
echo ""
echo "📦 Installing dependencies (this may take a while)..."
pip install -r requirements.txt

# Check Ollama
echo ""
echo "🦙 Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found."
    echo "   Install from: https://ollama.ai"
    echo "   Or run: brew install ollama"
else
    echo "✅ Ollama found"
    
    # Check if ollama is running
    if ! pgrep -x "ollama" > /dev/null; then
        echo "   Starting Ollama..."
        ollama serve &
        sleep 2
    fi
    
    # Pull recommended model
    echo "📥 Pulling Gemma 3 model (this may take a while)..."
    ollama pull gemma3:4b || echo "   ⚠️ Could not pull model. Run 'ollama pull gemma3:4b' manually."
fi

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p outputs temp models

# Done!
echo ""
echo "========================================"
echo "  ✅ Setup Complete!"
echo "========================================"
echo ""
echo "To start AutoKatha:"
echo ""
echo "  1. Activate the environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Make sure Ollama is running:"
echo "     ollama serve"
echo ""
echo "  3. Launch the app:"
echo "     python app.py"
echo ""
echo "  4. Open in browser:"
echo "     http://127.0.0.1:7860"
echo ""
echo "========================================"
