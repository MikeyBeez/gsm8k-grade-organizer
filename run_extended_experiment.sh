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
mkdir -p models/distilbert_original_extended
mkdir -p models/distilbert_ordered_extended
mkdir -p results

# Check requirements
check_requirements

# Run analysis first
echo "=== Step 0: Analyzing grade distribution ==="
python src/analyze_grade_distribution.py

# Step 1: Fine-tune on original dataset with extended training
echo "=== Step 1: Fine-tuning on original dataset (extended) ==="
python src/finetune_original.py --max-samples 1000 --num-epochs 10 --train-batch-size 8 --learning-rate 1e-5 --output-dir models/distilbert_original_extended

# Check if model was created
if [ ! -f "models/distilbert_original_extended/config.json" ]; then
    echo "Error: Original model not created. Skipping remaining steps."
    exit 1
fi

# Step 2: Fine-tune on ordered dataset with extended training
echo "=== Step 2: Fine-tuning on ordered dataset (extended) ==="
python src/finetune_ordered.py --max-samples 1000 --num-epochs 10 --train-batch-size 8 --learning-rate 1e-5 --output-dir models/distilbert_ordered_extended

# Check if model was created
if [ ! -f "models/distilbert_ordered_extended/config.json" ]; then
    echo "Error: Ordered model not created. Skipping remaining steps."
    exit 1
fi

# Step 3: Benchmark the original model
echo "=== Step 3: Benchmarking original model (extended) ==="
python src/benchmark.py --model-path models/distilbert_original_extended --output results/original_extended_benchmark.json --max-samples 100

# Step 4: Benchmark the ordered model
echo "=== Step 4: Benchmarking ordered model (extended) ==="
python src/benchmark.py --model-path models/distilbert_ordered_extended --output results/ordered_extended_benchmark.json --max-samples 100

# Step 5: Benchmark by grade level
echo "=== Step 5: Benchmarking by grade level (extended) ==="
python src/benchmark.py --model-path models/distilbert_original_extended --by-grade --max-samples 30 --output results/original_extended_by_grade.json
python src/benchmark.py --model-path models/distilbert_ordered_extended --by-grade --max-samples 30 --output results/ordered_extended_by_grade.json

# Step 6: Compare results
echo "=== Step 6: Comparing extended results ==="
python src/compare_results.py --original-results results/original_extended_benchmark.json --ordered-results results/ordered_extended_benchmark.json --original-by-grade results/original_extended_by_grade.json --ordered-by-grade results/ordered_extended_by_grade.json

echo "Extended experiment complete!"
