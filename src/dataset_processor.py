import json
import os
import time
from datasets import load_dataset
from tqdm import tqdm
from .grade_classifier import estimate_grade_level, query_deepseek, estimate_grade_by_heuristics
try:
    from .heuristic_classifier import classify_by_heuristics
except ImportError:
    # Fallback if the file doesn't exist yet
    classify_by_heuristics = estimate_grade_by_heuristics

class GSM8KProcessor:
    def __init__(self, output_dir="outputs", model="deepseek-r1", use_heuristics_only=False):
        self.output_dir = output_dir
        self.model = model
        self.use_heuristics_only = use_heuristics_only
        os.makedirs(output_dir, exist_ok=True)
    
    def load_dataset(self):
        print("Loading GSM8K dataset...")
        return load_dataset("gsm8k", "main")
    
    def organize_by_grade(self, dataset, sample_size=None):
        if self.use_heuristics_only:
            print("Processing GSM8K dataset using heuristics only...")
        else:
            print(f"Processing GSM8K dataset using {self.model} with heuristic fallback...")
        
        # Store problems by grade level
        graded_problems = {
            0: [],  # Kindergarten
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: []
        }
        
        # Process each problem in the training set
        subset = dataset["train"]
        if sample_size:
            subset = subset.select(range(min(sample_size, len(subset))))
        
        # Create progress bar
        pbar = tqdm(total=len(subset), desc="Classifying problems")
        
        for i, item in enumerate(subset):
            try:
                question = item["question"]
                answer = item["answer"]
                
                if self.use_heuristics_only:
                    # Use only heuristics
                    grade_level = classify_by_heuristics(question)
                else:
                    # Try using the LLM first
                    try:
                        # Create a prompt for model to analyze
                        prompt = estimate_grade_level(question)
                        
                        # Query model to determine the grade level
                        grade_level = query_deepseek(prompt, self.model)
                    except Exception as e:
                        print(f"Error with LLM, falling back to heuristics: {e}")
                        grade_level = classify_by_heuristics(question)
                
                # Store the problem with its grade level
                graded_problems[grade_level].append({
                    "id": i,
                    "question": question,
                    "answer": answer,
                    "grade_level": grade_level
                })
                
            except Exception as e:
                print(f"Error processing problem {i}: {e}")
                # Default to grade 4 for errors
                graded_problems[4].append({
                    "id": i,
                    "question": item["question"],
                    "answer": item["answer"],
                    "grade_level": 4,
                    "error": str(e)
                })
            
            # Update progress bar regardless of success/failure
            pbar.update(1)
            
            # Add a small delay to avoid overloading Ollama
            if not self.use_heuristics_only:
                time.sleep(0.2)
        
        # Close progress bar
        pbar.close()
        
        # Create a sorted version of the dataset
        sorted_dataset = []
        for grade in range(9):  # 0 (K) through 8
            sorted_dataset.extend(graded_problems[grade])
        
        return sorted_dataset, graded_problems
    
    def save_organized_dataset(self, sorted_dataset, graded_problems):
        # Save the combined organized dataset
        with open(f"{self.output_dir}/gsm8k_by_grade.json", "w") as f:
            json.dump(sorted_dataset, f, indent=2)
        
        # Create separate files for each grade level
        for grade in range(9):
            grade_name = "K" if grade == 0 else str(grade)
            with open(f"{self.output_dir}/grade_{grade_name}.json", "w") as f:
                json.dump(graded_problems[grade], f, indent=2)
        
        print(f"Dataset organized and saved to {self.output_dir}/")
        
        # Print statistics
        print("\nStatistics by grade level:")
        for grade in range(9):
            grade_name = "K" if grade == 0 else str(grade)
            print(f"Grade {grade_name}: {len(graded_problems[grade])} problems")
