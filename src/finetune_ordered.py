import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import json
import numpy as np
import re
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

def extract_answer(answer_text):
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

def load_ordered_dataset(json_path):
    """Load the ordered dataset from the JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to the format expected by the model
    dataset = {
        "question": [],
        "answer": [],
        "grade_level": []
    }
    
    for item in data:
        dataset["question"].append(item["question"])
        dataset["answer"].append(item["answer"])
        dataset["grade_level"].append(item["grade_level"])
    
    return dataset

def preprocess_function(examples):
    """Preprocess examples for the model"""
    # Extract questions
    questions = examples["question"]
    
    # Extract answers and convert to numerical values
    answers = []
    for ans in examples["answer"]:
        # Extract the numerical answer
        numeric_answer = extract_answer(ans)
        if numeric_answer is not None:
            answers.append(float(numeric_answer))
        else:
            answers.append(0.0)  # Default if parsing fails
    
    # Tokenize the questions
    tokenized_inputs = tokenizer(
        questions,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Add the labels (answers)
    tokenized_inputs["labels"] = torch.tensor(answers, dtype=torch.float).unsqueeze(1)
    
    return tokenized_inputs

def compute_metrics(eval_pred):
    """Compute metrics for regression task"""
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Calculate various metrics
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    
    # Calculate relative error
    rel_errors = np.abs(predictions - labels) / np.maximum(1.0, np.abs(labels))
    mean_rel_error = np.mean(rel_errors)
    
    # Calculate accuracy within 5% tolerance
    accuracy = np.mean(rel_errors < 0.05)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mean_rel_error": mean_rel_error,
        "accuracy": accuracy
    }

def main():
    import argparse
    from datasets import Dataset, DatasetDict
    
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on the ordered GSM8K dataset")
    parser.add_argument("--ordered-data", type=str, default="outputs/gsm8k_by_grade.json", 
                       help="Path to the ordered dataset JSON file")
    parser.add_argument("--output-dir", type=str, default="models/distilbert_ordered", 
                       help="Directory to save the model")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--test-split", type=float, default=0.1, 
                       help="Portion of data to use for testing (0.0-1.0)")
    parser.add_argument("--max-samples", type=int, default=None, 
                       help="Maximum number of samples to use (for faster testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ordered dataset
    print(f"Loading ordered dataset from {args.ordered_data}...")
    ordered_data = load_ordered_dataset(args.ordered_data)
    
    # Limit the number of samples if specified
    if args.max_samples and args.max_samples < len(ordered_data["question"]):
        for key in ordered_data.keys():
            ordered_data[key] = ordered_data[key][:args.max_samples]
    
    # Split into train and test
    test_size = int(len(ordered_data["question"]) * args.test_split)
    train_size = len(ordered_data["question"]) - test_size
    
    train_data = {key: values[:train_size] for key, values in ordered_data.items()}
    test_data = {key: values[train_size:] for key, values in ordered_data.items()}
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    
    print(f"Training on {len(train_dataset)} examples, evaluating on {len(test_dataset)} examples")
    
    # Initialize tokenizer
    global tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Preprocess the datasets
    print("Preprocessing training data...")
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    print("Preprocessing evaluation data...")
    tokenized_eval = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Initialize model
    print("Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=1  # Regression task
    )
    
    # Set up training arguments with minimal parameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print(f"Starting training for {args.num_epochs} epochs...")
    trainer.train()
    
    # Save the model
    print(f"Training complete. Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate the model
    print("Evaluating the model...")
    metrics = trainer.evaluate()
    
    print("\n============ Evaluation Results ============")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Model and metrics saved to {args.output_dir}")

if __name__ == "__main__":
    main()
