import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import re
import os
import random

class GSM8KBenchmark:
    def __init__(self, model_path, device=None):
        """Initialize the benchmark with a model path"""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model from {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_answer(self, answer_text):
        """Extract the final numerical answer from GSM8K answer text"""
        # GSM8K answers typically end with the final answer in format "The answer is X" or similar
        # Try to extract just the numeric value
        lines = answer_text.strip().split('\n')
        last_line = lines[-1]
        
        # Look for digits in the last line
        numbers = re.findall(r'\d+', last_line)
        if numbers:
            return float(numbers[-1])  # Return the last number found
        
        # If no number found, try alternative extraction methods
        for line in reversed(lines):
            numbers = re.findall(r'\d+', line)
            if numbers:
                return float(numbers[-1])
        
        # Fallback
        return None
    
    def predict_answer(self, question):
        """Use the model to predict an answer to a math question"""
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted value
        prediction = outputs.logits.item()
        return prediction
    
    def evaluate_test_set(self, test_data, max_samples=None):
        """Evaluate the model on the test set"""
        if isinstance(test_data, str):
            # Load from file
            with open(test_data, 'r') as f:
                test_data = json.load(f)
        
        # Convert Dataset to list if needed
        if hasattr(test_data, 'to_list'):
            test_data = test_data.to_list()
        
        if max_samples and max_samples < len(test_data):
            # Use a random subset for evaluation
            sample_data = test_data.copy()  # Create a copy first
            random.shuffle(sample_data)
            test_data = sample_data[:max_samples]
        
        results = []
        correct_count = 0
        total_count = 0
        
        for item in tqdm(test_data, desc="Evaluating"):
            question = item["question"]
            correct_answer = self.extract_answer(item["answer"])
            
            if correct_answer is not None:
                predicted_answer = self.predict_answer(question)
                
                # Calculate error
                error = abs(predicted_answer - correct_answer)
                relative_error = error / max(1, abs(correct_answer))  # Avoid division by zero
                
                # Consider it correct if the relative error is less than 5%
                is_correct = relative_error < 0.05
                
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                results.append({
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "error": error,
                    "relative_error": relative_error,
                    "is_correct": is_correct
                })
        
        # Calculate accuracy
        accuracy = correct_count / max(1, total_count)
        
        # Calculate average error
        avg_error = np.mean([r["error"] for r in results])
        avg_relative_error = np.mean([r["relative_error"] for r in results])
        
        # Return metrics and detailed results
        metrics = {
            "accuracy": accuracy,
            "avg_error": avg_error,
            "avg_relative_error": avg_relative_error,
            "correct_count": correct_count,
            "total_count": total_count
        }
        
        return metrics, results
    
    def evaluate_by_grade(self, base_path="outputs", max_samples_per_grade=50):
        """Evaluate performance by grade level"""
        grade_metrics = {}
        
        for grade in range(9):  # K through 8
            grade_name = "K" if grade == 0 else str(grade)
            file_path = os.path.join(base_path, f"grade_{grade_name}.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    grade_data = json.load(f)
                
                if not grade_data:
                    grade_metrics[grade_name] = {"error": "No data"}
                    continue
                
                print(f"Evaluating Grade {grade_name} ({len(grade_data)} problems)")
                metrics, _ = self.evaluate_test_set(grade_data, max_samples=max_samples_per_grade)
                grade_metrics[grade_name] = metrics
            else:
                grade_metrics[grade_name] = {"error": "File not found"}
        
        return grade_metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark a model on GSM8K")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test-data", type=str, default=None, help="Path to test data JSON file (optional)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--by-grade", action="store_true", help="Evaluate performance by grade level")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    benchmark = GSM8KBenchmark(args.model_path)
    
    if args.by_grade:
        # Evaluate by grade level
        metrics = benchmark.evaluate_by_grade(max_samples_per_grade=args.max_samples)
        
        # Print results
        print("\n============ Results by Grade ============")
        for grade, grade_metrics in metrics.items():
            if "accuracy" in grade_metrics:
                print(f"Grade {grade}: Accuracy = {grade_metrics['accuracy']:.2%}, Avg Error = {grade_metrics['avg_error']:.2f}")
            else:
                print(f"Grade {grade}: {grade_metrics.get('error', 'Unknown error')}")
                
        # Save results
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Grade-level results saved to {args.output}")
    else:
        # Load test data
        if args.test_data:
            test_data = args.test_data
        else:
            # Use the default GSM8K test set
            dataset = load_dataset("gsm8k", "main")
            test_data = dataset["test"]
        
        # Run evaluation
        metrics, results = benchmark.evaluate_test_set(test_data, max_samples=args.max_samples)
        
        # Print results
        print("\n============ Benchmark Results ============")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Average Error: {metrics['avg_error']:.2f}")
        print(f"Average Relative Error: {metrics['avg_relative_error']:.2%}")
        print(f"Correct: {metrics['correct_count']} / {metrics['total_count']}")
        
        # Save detailed results
        output = {
            "metrics": metrics,
            "results": results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()
