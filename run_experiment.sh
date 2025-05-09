#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check for required packages
check_requirements() {
    echo "Checking requirements..."
    
    # Check for Python
    if ! command_exists python; then
        echo "Error: Python not found"
        exit 1
    fi
    
    # Install required Python packages
    echo "Installing required packages..."
    uv pip install torch transformers datasets matplotlib scikit-learn accelerate safetensors
    
    echo "All requirements satisfied."
}

# Create directories
mkdir -p models/distilbert_original
mkdir -p models/distilbert_ordered
mkdir -p results

# Check requirements
check_requirements

# Run analysis first
echo "=== Step 0: Analyzing grade distribution ==="
python src/analyze_grade_distribution.py

# Step 1: Fine-tune on original dataset with longer training
echo "=== Step 1: Fine-tuning on original dataset ==="
python src/finetune_original.py --max-samples 200 --num-epochs 5 --train-batch-size 8 --learning-rate 1e-5

# Check if model was created
if [ ! -f "models/distilbert_original/config.json" ]; then
    echo "Error: Original model not created. Skipping remaining steps."
    exit 1
fi

# Step 2: Fine-tune on ordered dataset with longer training
echo "=== Step 2: Fine-tuning on ordered dataset ==="
python src/finetune_ordered.py --max-samples 200 --num-epochs 5 --train-batch-size 8 --learning-rate 1e-5

# Check if model was created
if [ ! -f "models/distilbert_ordered/config.json" ]; then
    echo "Error: Ordered model not created. Skipping remaining steps."
    exit 1
fi

# Step 3: Benchmark the original model
echo "=== Step 3: Benchmarking original model ==="
python src/benchmark.py --model-path models/distilbert_original --output results/original_benchmark.json --max-samples 50

# Step 4: Benchmark the ordered model
echo "=== Step 4: Benchmarking ordered model ==="
python src/benchmark.py --model-path models/distilbert_ordered --output results/ordered_benchmark.json --max-samples 50

# Step 5: Benchmark by grade level (just K-2 since that's where most problems are)
echo "=== Step 5: Benchmarking by grade level ==="
python src/benchmark.py --model-path models/distilbert_original --by-grade --max-samples 20 --output results/original_by_grade.json
python src/benchmark.py --model-path models/distilbert_ordered --by-grade --max-samples 20 --output results/ordered_by_grade.json

# Step 6: Compare results
echo "=== Step 6: Comparing results ==="
python src/compare_results.py

echo "Experiment complete!"
