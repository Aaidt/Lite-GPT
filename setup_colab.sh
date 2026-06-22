#!/bin/bash
set -e

echo "cloning repo..."
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
cd src/data/datasets/ && ls && cd ../..

cat << EOF

To log into wandb run:

import wandb
wandb.login(key="YOUR_API_KEY")

Run training:

uv run python -m src.training.train

EOF


echo "setup complete!!!"