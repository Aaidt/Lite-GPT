#!/bin/bash
set -e

echo "cloning repo..."
rm -rf Lite-GPT
git clone https://github.com/Aaidt/Lite-GPT.git

cd Lite-GPT

echo "creating necessary folders needed..."
mkdir -p checkpoints results logs

echo "installing uv and other packages..."
pip install -q uv 

echo "Installing dependencies..."
uv sync

echo "Installing the shakespeare dataset..."
python -m src.litegpt_16M.data.datasets

echo "Tokenizing the dataset..."
python -m src.litegpt_16M.data.tokenizer

echo "check installed dataset and tokens..."
ls src/litegpt_16M/data/datasets/
ls src/litegpt_16M/data/datasets/tokens 

echo ""
read -p "Start training? [y/N]: " choice

case "$choice" in
    [yY]|[yY][eE][sS])
        echo "Starting training..."
        python -m src.litegpt_16M.training.train
        ;;
    *)
        echo "Setup complete. Training not started."
        ;;
esac 