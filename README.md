# GSM8K Grade Organizer

This project organizes the GSM8K (Grade School Math 8K) dataset by grade level using DeepSeek through Ollama.

## Features

- Classifies math problems from the GSM8K dataset by US grade level (K-8)
- Uses DeepSeek LLM through Ollama for accurate grade classification
- Organizes problems in ascending order of difficulty
- Saves organized datasets as JSON files

## Requirements

- Python 3.8+
- Ollama with DeepSeek model installed
- Required Python packages (installed via uv)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gsm8k-grade-organizer.git
   cd gsm8k-grade-organizer
   ```

2. Set up a virtual environment (if not already created):
   ```bash
   uv venv
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Ensure Ollama is installed and DeepSeek model is available:
   ```bash
   ollama list  # Check if deepseek-r1 is installed
   # If not installed, run:
   # ollama pull deepseek-r1
   ```

## Usage

Run the main script to process and organize the GSM8K dataset:

```bash
python main.py --output outputs --model deepseek-r1 --sample 100
```

Options:
- `--output`: Output directory (default: "outputs")
- `--model`: Ollama model to use (default: "deepseek-r1")
- `--sample`: Process a sample of N problems (default: all)

## Output

The script generates:
- A combined JSON file with all problems sorted by grade level
- Separate JSON files for each grade level (K-8)
- Statistics on problem distribution by grade

## How It Works

The project uses a Language Model (DeepSeek through Ollama) to analyze each math problem and determine its grade level based on:

1. Mathematical concepts present in the problem
2. Vocabulary complexity
3. Number of steps required to solve
4. Type of operations involved

The LLM analyzes each problem and assigns a grade level from Kindergarten (K) to Grade 8.

## Example

Original problem: "Janet has 5 apples. She gives 2 to her friend. How many does she have left?"
- Classified as: Grade K (Kindergarten)

Original problem: "If a rectangle has a length of 8 cm and a width of 6 cm, what is its area in square centimeters?"
- Classified as: Grade 3

## License

MIT

## Repository Structure

This repository contains the essential code to reproduce our curriculum learning experiments:

- `src/`: Source code for the experiments
- `outputs/`: Directory where the GSM8K dataset organized by grade level will be stored
- `results/`: Visualizations and result files
- `run_experiment.sh`: Script to run the baseline experiment
- `run_extended_experiment.sh`: Script to run the extended experiment

## Data Files

The JSON data files are not included in the repository due to their size, but will be generated when you run the experiments.

## Reproducibility

To reproduce our results:

1. Clone this repository
2. Install dependencies: `uv pip install torch transformers datasets matplotlib scikit-learn accelerate safetensors`
3. Run the baseline experiment: `./run_experiment.sh`
4. Run the extended experiment: `./run_extended_experiment.sh`

The scripts will automatically:
- Download the GSM8K dataset
- Process it and organize it by grade level
- Train models on the original and ordered datasets
- Evaluate performance and generate visualizations
