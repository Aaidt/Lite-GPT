#!/bin/bash
set -e

if [ -d "Lite-GPT" ] && [ -d "Lite-GPT/.git" ]; then
    echo "repo exists, pulling latest changes..."
    cd Lite-GPT
    git pull
else
    echo "cloning repo..."
    rm -rf Lite-GPT
    git clone https://github.com/Aaidt/Lite-GPT.git
    cd Lite-GPT
fi

echo "creating folders for logging and checkpointing..."
mkdir -p checkpoints results logs

echo "installing uv and other packages..."
pip install -q uv 

echo "Installing dependencies..."
uv sync

echo "Installing the 500M token dataset..."
python -m src.litegpt_25M.data.datasets

echo "check installed token files   ..."
ls src/litegpt_25M/data/datasets/tokens 

echo ""
read -p "Start training? [y/N]: " choice

case "$choice" in
    [yY]|[yY][eE][sS])
        echo "Starting training..."
        python -m src.litegpt_25M.training.train
        ;;
    *)
        echo "Setup complete. Training not started."
        ;;
esac 