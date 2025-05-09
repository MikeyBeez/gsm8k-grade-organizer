import argparse
import os
from src.dataset_processor import GSM8KProcessor

def main():
    parser = argparse.ArgumentParser(description="GSM8K Grade Level Organizer")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="deepseek-r1", help="Ollama model to use")
    parser.add_argument("--sample", type=int, default=None, help="Process a sample of N problems (default: all)")
    parser.add_argument("--heuristic-only", action="store_true", help="Use only heuristics, no LLM")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize processor
    processor = GSM8KProcessor(output_dir=args.output, model=args.model, use_heuristics_only=args.heuristic_only)
    
    # Load dataset
    dataset = processor.load_dataset()
    
    # Process and organize
    sorted_dataset, graded_problems = processor.organize_by_grade(dataset, sample_size=args.sample)
    
    # Save results
    processor.save_organized_dataset(sorted_dataset, graded_problems)
    
    print("GSM8K dataset organization complete!")


if __name__ == "__main__":
    main()
