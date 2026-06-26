#!/bin/bash
set -e

echo ""
echo "██╗     ██╗████████╗███████╗     ██████╗ ██████╗ ████████╗"
echo "██║     ██║╚══██╔══╝██╔════╝    ██╔════╝ ██╔══██╗╚══██╔══╝"
echo "██║     ██║   ██║   █████╗      ██║  ███╗██████╔╝   ██║   "
echo "██║     ██║   ██║   ██╔══╝      ██║   ██║██╔═══╝    ██║   "
echo "███████╗██║   ██║   ███████╗    ╚██████╔╝██║        ██║   "
echo ""

if [ -d "Lite-GPT" ] && [ -d "Lite-GPT/.git" ]; then
   echo "📂  Repository exists, pulling latest changes..."
   cd Lite-GPT
   git pull
else
   echo "📥  Cloning repository..."
   git clone https://github.com/Aaidt/Lite-GPT.git
   cd Lite-GPT
fi

if [ -f /content/train.bin ] && [ -f /content/val.bin ]; then
   echo "📦  Found pre-existing token files in /content, moving them to datasets/tokens..."
   mkdir -p src/litegpt_25M/data/datasets/tokens
   mv /content/train.bin src/litegpt_25M/data/datasets/tokens/
   mv /content/val.bin src/litegpt_25M/data/datasets/tokens/
fi

echo "📦  Installing uv and other packages..."
pip install -q uv

echo "📦  Installing dependencies..."
uv sync
uv pip install setuptools

echo "📦  Installing the 500M token dataset..."
uv run python -m src.litegpt_25M.data.datasets

echo "🔍  Checking installed token files..."
ls src/litegpt_25M/data/datasets/tokens

echo ""
echo "──────────────────────────────────────────────"
read -p "  🎯  Start training? [y/N]: " choice

case "$choice" in
[yY] | [yY][eE][sS])
   echo ""
   echo "══════════════════════════════════════════"
   echo "         🚀  Starting training...         "
   echo "══════════════════════════════════════════"
   echo ""
   uv run python -m src.litegpt_25M.training.train
   ;;
*)
   echo ""
   echo "✅  Setup complete. Training not started."
   ;;
esac