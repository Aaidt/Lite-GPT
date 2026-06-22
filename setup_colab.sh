#!/bin/bash
set -e

echo "creating necessary folders needed..."
mkdir -p checkpoints results logs

echo "cloning repo..."
rm -rf Lite-GPT
git clone https://github.com/Aaidt/Lite-GPT.git

echo "moving checkpoints/results/logs files into the repo..."
for dir in checkpoints results logs; do
    [ -d "$dir" ] && mv "$dir" Lite-GPT/
done

cd Lite-GPT

echo "installing uv and other packages..."
pip install -q uv 

echo "Installing dependencies..."
uv sync

echo "Installing the shakespeare dataset..."
python -m src.data.datasets

echo "Tokenizing the dataset..."
python -m src.data.tokenizer

echo "check installed dataset and tokens..."
ls src/data/datasets/
ls src/data/datasets/tokens 

echo ""
read -p "Start training? [y/N]: " choice

case "$choice" in
    [yY]|[yY][eE][sS])
        echo "Starting training..."
        python -m src.training.train
        ;;
    *)
        echo "Setup complete. Training not started."
        ;;
esac