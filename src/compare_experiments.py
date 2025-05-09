import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        print(f"Error loading {file_path}")
        return None

def compare_experiments():
    # Load results from both experiments
    original_baseline = load_results("results/original_benchmark.json")
    ordered_baseline = load_results("results/ordered_benchmark.json")
    original_extended = load_results("results/original_extended_benchmark.json")
    ordered_extended = load_results("results/ordered_extended_benchmark.json")
    
    if not all([original_baseline, ordered_baseline, original_extended, ordered_extended]):
        print("Some result files are missing. Run both experiments first.")
        return
    
    # Extract metrics
    metrics = ["accuracy", "avg_error", "avg_relative_error"]
    models = ["Original (Baseline)", "Ordered (Baseline)", "Original (Extended)", "Ordered (Extended)"]
    
    values = {
        "accuracy": [
            original_baseline["metrics"]["accuracy"],
            ordered_baseline["metrics"]["accuracy"],
            original_extended["metrics"]["accuracy"],
            ordered_extended["metrics"]["accuracy"]
        ],
        "avg_error": [
            original_baseline["metrics"]["avg_error"],
            ordered_baseline["metrics"]["avg_error"],
            original_extended["metrics"]["avg_error"],
            ordered_extended["metrics"]["avg_error"]
        ],
        "avg_relative_error": [
            original_baseline["metrics"]["avg_relative_error"],
            ordered_baseline["metrics"]["avg_relative_error"],
            original_extended["metrics"]["avg_relative_error"],
            ordered_extended["metrics"]["avg_relative_error"]
        ]
    }
    
    # Print comparison
    print("\n===== Experiment Comparison =====")
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for i, model in enumerate(models):
            print(f"  {model}: {values[metric][i]:.4f}")
        
        # Calculate improvements
        baseline_improvement = ((values[metric][1] - values[metric][0]) / values[metric][0]) * 100
        extended_improvement = ((values[metric][3] - values[metric][2]) / values[metric][2]) * 100
        
        if metric == "accuracy":
            # For accuracy, higher is better
            print(f"  Baseline improvement: {baseline_improvement:+.2f}%")
            print(f"  Extended improvement: {extended_improvement:+.2f}%")
        else:
            # For errors, lower is better
            print(f"  Baseline improvement: {-baseline_improvement:+.2f}%")
            print(f"  Extended improvement: {-extended_improvement:+.2f}%")
    
    # Create visualization for accuracy (bar chart)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.8
    
    plt.bar(x, values["accuracy"], width)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, models, rotation=15)
    
    # Add value labels
    for i, v in enumerate(values["accuracy"]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png')
    
    # Create visualization for errors (log scale)
    plt.figure(figsize=(12, 6))
    
    # Process error values (log scale works better for large differences)
    error_values = values["avg_error"]
    
    plt.bar(x, error_values, width)
    plt.ylabel('Average Error (log scale)')
    plt.yscale('log')  # Use log scale for better visualization
    plt.title('Model Error Comparison')
    plt.xticks(x, models, rotation=15)
    
    # Add value labels
    for i, v in enumerate(error_values):
        plt.text(i, v * 1.1, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('results/error_comparison.png')
    
    print("\nComparison visualizations saved to results/accuracy_comparison.png and results/error_comparison.png")

if __name__ == "__main__":
    compare_experiments()
