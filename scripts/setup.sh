#!/bin/bash
# Quick setup script for TrajectoryRL
set -e

echo "========================================"
echo "TrajectoryRL Setup"
echo "========================================"
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher required (found $python_version)"
    exit 1
fi

echo "✓ Python $python_version"

# Check for ClawBench
if [ ! -d "./clawbench" ]; then
    echo "⚠️  ClawBench not found in ./clawbench"
    echo "   Clone it: git clone https://github.com/trajectoryRL/clawbench.git ./clawbench"
    exit 1
fi

echo "✓ ClawBench found"

# Create .env if missing
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your API keys and wallet name"
fi

echo "✓ .env exists"

# Install package
echo
echo "Installing TrajectoryRL..."
pip install -e .

echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. Run validator: python neurons/validator.py"
echo "  3. Or run with Docker: cd docker && docker compose up"
echo
