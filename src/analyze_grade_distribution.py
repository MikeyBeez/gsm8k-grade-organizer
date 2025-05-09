import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_data(json_path):
    """Load the organized dataset from the JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_grade_distribution(data):
    """Analyze the distribution of problems by grade level"""
    # Count problems by grade level
    grade_counts = Counter()
    
    for item in data:
        grade_level = item.get("grade_level")
        if grade_level is not None:
            # Convert to string representation (K, 1, 2, etc.)
            grade_str = "K" if grade_level == 0 else str(grade_level)
            grade_counts[grade_str] += 1
    
    # Print the counts
    print("\n=== Grade Level Distribution ===")
    total = sum(grade_counts.values())
    
    for grade in sorted(grade_counts.keys(), key=lambda x: 0 if x == 'K' else int(x)):
        count = grade_counts[grade]
        percentage = (count / total) * 100
        print(f"Grade {grade}: {count} problems ({percentage:.1f}%)")
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    
    # Sort grades properly (K comes before 1)
    sorted_grades = sorted(grade_counts.keys(), key=lambda x: 0 if x == 'K' else int(x))
    counts = [grade_counts[g] for g in sorted_grades]
    
    bars = plt.bar(sorted_grades, counts)
    
    plt.title('Distribution of Problems by Grade Level')
    plt.xlabel('Grade Level')
    plt.ylabel('Number of Problems')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/grade_distribution.png')
    print("Grade distribution chart saved to results/grade_distribution.png")
    
    return grade_counts

def analyze_problem_complexity(data):
    """Analyze problem complexity by grade level"""
    grade_word_counts = {}
    
    for item in data:
        grade_level = item.get("grade_level")
        question = item.get("question", "")
        
        if grade_level is not None and question:
            # Convert to string representation
            grade_str = "K" if grade_level == 0 else str(grade_level)
            
            # Count words
            word_count = len(question.split())
            
            if grade_str not in grade_word_counts:
                grade_word_counts[grade_str] = []
            
            grade_word_counts[grade_str].append(word_count)
    
    # Calculate average word count by grade
    avg_word_counts = {}
    for grade, counts in grade_word_counts.items():
        avg_word_counts[grade] = sum(counts) / len(counts)
    
    # Print the results
    print("\n=== Average Problem Length by Grade ===")
    
    for grade in sorted(avg_word_counts.keys(), key=lambda x: 0 if x == 'K' else int(x)):
        avg = avg_word_counts[grade]
        print(f"Grade {grade}: {avg:.1f} words")
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    
    # Sort grades properly
    sorted_grades = sorted(avg_word_counts.keys(), key=lambda x: 0 if x == 'K' else int(x))
    avgs = [avg_word_counts[g] for g in sorted_grades]
    
    bars = plt.bar(sorted_grades, avgs)
    
    plt.title('Average Problem Length by Grade Level')
    plt.xlabel('Grade Level')
    plt.ylabel('Average Word Count')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/problem_length_by_grade.png')
    print("Problem length chart saved to results/problem_length_by_grade.png")
    
    return avg_word_counts

def analyze_numerical_complexity(data):
    """Analyze numerical complexity by grade level"""
    grade_num_counts = {}
    
    for item in data:
        grade_level = item.get("grade_level")
        question = item.get("question", "")
        
        if grade_level is not None and question:
            # Convert to string representation
            grade_str = "K" if grade_level == 0 else str(grade_level)
            
            # Count numbers in the question
            import re
            numbers = re.findall(r'\d+', question)
            max_num = 0 if not numbers else max([int(n) for n in numbers])
            
            if grade_str not in grade_num_counts:
                grade_num_counts[grade_str] = []
            
            grade_num_counts[grade_str].append(max_num)
    
    # Calculate average max number by grade
    avg_max_num = {}
    for grade, nums in grade_num_counts.items():
        avg_max_num[grade] = sum(nums) / len(nums)
    
    # Print the results
    print("\n=== Average Maximum Number by Grade ===")
    
    for grade in sorted(avg_max_num.keys(), key=lambda x: 0 if x == 'K' else int(x)):
        avg = avg_max_num[grade]
        print(f"Grade {grade}: {avg:.1f}")
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    
    # Sort grades properly
    sorted_grades = sorted(avg_max_num.keys(), key=lambda x: 0 if x == 'K' else int(x))
    avgs = [avg_max_num[g] for g in sorted_grades]
    
    bars = plt.bar(sorted_grades, avgs)
    
    plt.title('Average Maximum Number by Grade Level')
    plt.xlabel('Grade Level')
    plt.ylabel('Average Maximum Number')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/number_size_by_grade.png')
    print("Number size chart saved to results/number_size_by_grade.png")
    
    return avg_max_num

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze the grade distribution of the GSM8K dataset")
    parser.add_argument("--data", type=str, default="outputs/gsm8k_by_grade.json", 
                        help="Path to the organized dataset JSON file")
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.data):
        print(f"Error: File not found: {args.data}")
        return
    
    # Load the data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    print(f"Loaded {len(data)} problems")
    
    # Analyze grade distribution
    grade_counts = analyze_grade_distribution(data)
    
    # Analyze problem complexity
    avg_word_counts = analyze_problem_complexity(data)
    
    # Analyze numerical complexity
    avg_max_num = analyze_numerical_complexity(data)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
