import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(file_path):
    """Load benchmark results from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_overall_performance(original_results, ordered_results):
    """Compare the overall performance of the two models"""
    if not original_results or not ordered_results:
        print("Cannot compare overall performance: missing results")
        return
    
    original_metrics = original_results.get("metrics", {})
    ordered_metrics = ordered_results.get("metrics", {})
    
    # Print comparison
    print("\n=== Overall Performance Comparison ===")
    for metric in ["accuracy", "avg_error", "avg_relative_error"]:
        if metric in original_metrics and metric in ordered_metrics:
            original_value = original_metrics[metric]
            ordered_value = ordered_metrics[metric]
            
            # Calculate improvement (handle division by zero)
            if original_value == 0:
                if ordered_value == 0:
                    improvement = 0
                else:
                    improvement = float('inf') if ordered_value > 0 else float('-inf')
            else:
                improvement = ((ordered_value - original_value) / original_value) * 100
            
            print(f"{metric.capitalize()}:")
            print(f"  Original model: {original_value:.4f}")
            print(f"  Ordered model:  {ordered_value:.4f}")
            print(f"  Improvement:    {improvement:+.2f}%" if abs(improvement) != float('inf') else f"  Improvement:    N/A (from zero)")
            print()
    
    # Create bar chart
    metrics_to_plot = ["accuracy"]
    if all(m in original_metrics and m in ordered_metrics for m in metrics_to_plot):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        original_values = [original_metrics[m] for m in metrics_to_plot]
        ordered_values = [ordered_metrics[m] for m in metrics_to_plot]
        
        rects1 = ax.bar(x - width/2, original_values, width, label='Original Dataset')
        rects2 = ax.bar(x + width/2, ordered_values, width, label='Ordered Dataset')
        
        ax.set_ylabel('Value')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics_to_plot])
        ax.legend()
        
        for rect in rects1:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        for rect in rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig('results/performance_comparison.png')
        print("Performance comparison chart saved to results/performance_comparison.png")

def compare_grade_performance(original_by_grade, ordered_by_grade):
    """Compare the performance by grade level"""
    if not original_by_grade or not ordered_by_grade:
        print("Cannot compare grade performance: missing results")
        return
    
    # Get common grades with accuracy metrics
    grades = []
    for g in ["K", "1", "2", "3", "4", "5", "6", "7", "8"]:
        if g in original_by_grade and g in ordered_by_grade:
            if "accuracy" in original_by_grade[g] and "accuracy" in ordered_by_grade[g]:
                grades.append(g)
    
    if not grades:
        print("No common grades found with accuracy metrics")
        return
    
    # Print comparison
    print("\n=== Performance Comparison by Grade ===")
    for grade in grades:
        original_acc = original_by_grade[grade]["accuracy"]
        ordered_acc = ordered_by_grade[grade]["accuracy"]
        
        # Handle division by zero in improvement calculation
        if original_acc == 0:
            if ordered_acc == 0:
                improvement = 0
            else:
                improvement = float('inf') if ordered_acc > 0 else float('-inf')
        else:
            improvement = ((ordered_acc - original_acc) / original_acc) * 100
        
        print(f"Grade {grade}:")
        print(f"  Original model: {original_acc:.4f}")
        print(f"  Ordered model:  {ordered_acc:.4f}")
        print(f"  Improvement:    {improvement:+.2f}%" if abs(improvement) != float('inf') else f"  Improvement:    N/A (from zero)")
        print()
    
    # Create bar chart for accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(grades))
    width = 0.35
    
    original_values = [original_by_grade[g]["accuracy"] for g in grades]
    ordered_values = [ordered_by_grade[g]["accuracy"] for g in grades]
    
    rects1 = ax.bar(x - width/2, original_values, width, label='Original Dataset')
    rects2 = ax.bar(x + width/2, ordered_values, width, label='Ordered Dataset')
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Grade Level')
    ax.set_title('Model Performance by Grade Level')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Grade {g}' for g in grades])
    ax.legend()
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/grade_performance_comparison.png')
    print("Grade performance comparison chart saved to results/grade_performance_comparison.png")
    
    # Create bar chart for error rates
    fig, ax = plt.subplots(figsize=(12, 6))
    
    original_errors = [original_by_grade[g].get("avg_error", 0) for g in grades]
    ordered_errors = [ordered_by_grade[g].get("avg_error", 0) for g in grades]
    
    # Cap very large errors for better visualization
    max_error_cap = 1000
    original_errors = [min(e, max_error_cap) for e in original_errors]
    ordered_errors = [min(e, max_error_cap) for e in ordered_errors]
    
    rects1 = ax.bar(x - width/2, original_errors, width, label='Original Dataset')
    rects2 = ax.bar(x + width/2, ordered_errors, width, label='Ordered Dataset')
    
    ax.set_ylabel('Average Error (capped at 1000)')
    ax.set_xlabel('Grade Level')
    ax.set_title('Model Error Rates by Grade Level')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Grade {g}' for g in grades])
    ax.legend()
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/grade_error_comparison.png')
    print("Grade error comparison chart saved to results/grade_error_comparison.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--original-results", type=str, default="results/original_benchmark.json",
                        help="Path to original model benchmark results")
    parser.add_argument("--ordered-results", type=str, default="results/ordered_benchmark.json",
                        help="Path to ordered model benchmark results")
    parser.add_argument("--original-by-grade", type=str, default="results/original_by_grade.json",
                        help="Path to original model by-grade results")
    parser.add_argument("--ordered-by-grade", type=str, default="results/ordered_by_grade.json",
                        help="Path to ordered model by-grade results")
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Compare overall performance
    original_results = load_results(args.original_results)
    ordered_results = load_results(args.ordered_results)
    compare_overall_performance(original_results, ordered_results)
    
    # Compare performance by grade
    original_by_grade = load_results(args.original_by_grade)
    ordered_by_grade = load_results(args.ordered_by_grade)
    compare_grade_performance(original_by_grade, ordered_by_grade)

if __name__ == "__main__":
    main()
