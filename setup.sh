#!/bin/bash

# Create directories
mkdir -p src
mkdir -p models/distilbert_original
mkdir -p models/distilbert_ordered
mkdir -p outputs
mkdir -p results
mkdir -p tests

# Install required packages
echo "Installing PyTorch and related packages..."
uv pip install torch
uv pip install transformers
uv pip install datasets
uv pip install scikit-learn matplotlib

echo "Setup complete!"
