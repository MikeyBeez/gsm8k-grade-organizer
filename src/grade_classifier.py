import re
import subprocess
import time
import os
import signal
from tqdm import tqdm

def estimate_grade_level(problem):
    # Create a prompt for DeepSeek to analyze
    prompt = f"""
    Analyze this math problem and determine its US grade level (K-8). Consider the mathematical concepts, 
    vocabulary, and complexity.
    
    Problem: {problem}
    
    Grade level concepts:
    - Kindergarten: Basic counting, simple addition/subtraction within 10
    - Grade 1: Addition/subtraction within 100, basic place value
    - Grade 2: Addition/subtraction within 1000, money, time, basic measurement
    - Grade 3: Multiplication/division, basic fractions, area, perimeter
    - Grade 4: Multi-digit operations, fraction equivalence, decimals
    - Grade 5: Operations with fractions/decimals, volume, order of operations
    - Grade 6: Ratios, percentages, basic equations, negative numbers
    - Grade 7: Proportional relationships, statistics, probability
    - Grade 8: Linear equations, functions, Pythagorean theorem
    
    Return only the grade level number (K, 1, 2, 3, 4, 5, 6, 7, or 8) with no explanation.
    """
    
    return prompt

def run_with_timeout(cmd, timeout_sec):
    """Run command with timeout"""
    start = time.time()
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    stdout_data = []
    stderr_data = []
    
    while process.poll() is None:
        # Check if we've exceeded the timeout
        if time.time() - start > timeout_sec:
            # Try to terminate the process gracefully
            process.terminate()
            time.sleep(0.5)
            
            # Force kill if it didn't terminate
            if process.poll() is None:
                process.kill()
                
            raise subprocess.TimeoutExpired(cmd, timeout_sec)
            
        # Read output if available
        if process.stdout:
            line = process.stdout.readline()
            if line:
                stdout_data.append(line)
                
        if process.stderr:
            line = process.stderr.readline()
            if line:
                stderr_data.append(line)
                
        # Small sleep to avoid hogging CPU
        time.sleep(0.1)
    
    # Get any remaining output
    if process.stdout:
        stdout_data.extend(process.stdout.readlines())
    if process.stderr:
        stderr_data.extend(process.stderr.readlines())
        
    stdout = ''.join(stdout_data)
    stderr = ''.join(stderr_data)
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
        
    return stdout, stderr

def query_deepseek(prompt, model="deepseek-r1", max_retries=3, timeout=45):
    """
    Query the Ollama model with retries and improved error handling
    """
    for attempt in range(max_retries):
        try:
            # Prepare the command
            cmd = ["ollama", "run", model, prompt]
            
            # Run with custom timeout handler
            stdout, stderr = run_with_timeout(cmd, timeout)
            
            # Extract just the grade level from the output
            output = stdout.strip()
            # Look for K or a number 1-8
            match = re.search(r'(K|[1-8])', output)
            if match:
                grade = match.group(0)
                # Convert K to 0 for sorting purposes
                if grade == 'K':
                    return 0
                else:
                    return int(grade)
            else:
                # If we can't find a grade level, use simple heuristics to estimate
                print(f"Could not determine grade from model output, using heuristics")
                return estimate_grade_by_heuristics(prompt)
                
        except subprocess.TimeoutExpired as e:
            print(f"Attempt {attempt+1}/{max_retries}: Timeout error when calling Ollama. Retrying...")
            time.sleep(2)  # Wait before retrying
            
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed with timeout. Using heuristics instead.")
                return estimate_grade_by_heuristics(prompt)
                
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries}: Error calling Ollama: {e}")
            time.sleep(2)  # Wait before retrying
            
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed. Using heuristics instead.")
                return estimate_grade_by_heuristics(prompt)

def estimate_grade_by_heuristics(prompt):
    """
    Fallback method to estimate grade level based on simple heuristics
    when the LLM fails to provide a valid response
    """
    # Extract the problem from the prompt
    match = re.search(r'Problem: (.*?)Grade level concepts:', prompt, re.DOTALL)
    if not match:
        return 4  # Default to grade 4 if we can't extract the problem
    
    problem = match.group(1).strip()
    
    # Simple keyword-based heuristics
    keywords = {
        0: ['counting', 'how many', 'add', 'take away', 'less than', 'more than', 'simple', 'basic'],
        1: ['tens', 'ones', 'place value', 'double', 'half', 'count by'],
        2: ['hundred', 'money', 'dollar', 'cent', 'time', 'clock', 'hour', 'minute'],
        3: ['multiply', 'divide', 'fraction', 'area', 'perimeter', 'array'],
        4: ['decimal', 'equivalent fraction', 'compare fraction', 'angle', 'degree'],
        5: ['add fraction', 'subtract fraction', 'multiply fraction', 'volume', 'operation'],
        6: ['ratio', 'percent', 'proportion', 'negative', 'equation', 'expression'],
        7: ['proportion', 'statistics', 'probability', 'chance', 'likely', 'scale factor'],
        8: ['function', 'linear', 'slope', 'pythagorean', 'theorem', 'scientific notation']
    }
    
    # Check for operations
    operations = {
        'addition': 1,
        'subtraction': 1,
        'multiplication': 3,
        'division': 3,
        'fraction': 4,
        'decimal': 4,
        'percentage': 6,
        'ratio': 6,
        'exponent': 8
    }
    
    # Count occurrences of keywords and operations
    grades = [0] * 9
    
    # Check keywords
    for grade, words in keywords.items():
        for word in words:
            if word.lower() in problem.lower():
                grades[grade] += 1
    
    # Check for operations by looking for symbols
    if '+' in problem or 'add' in problem.lower() or 'sum' in problem.lower():
        grades[1] += 1
    if '-' in problem or 'subtract' in problem.lower() or 'difference' in problem.lower():
        grades[1] += 1
    if '*' in problem or 'x' in problem or 'multiply' in problem.lower() or 'product' in problem.lower():
        grades[3] += 1
    if '/' in problem or 'divide' in problem.lower() or 'quotient' in problem.lower():
        grades[3] += 1
    if '%' in problem or 'percent' in problem.lower():
        grades[6] += 1
    
    # Assess complexity
    words = problem.split()
    if len(words) < 15:
        complexity_grade = 2
    elif len(words) < 25:
        complexity_grade = 4
    elif len(words) < 40:
        complexity_grade = 6
    else:
        complexity_grade = 8
    
    grades[complexity_grade] += 1
    
    # Determine the most likely grade
    max_grade = grades.index(max(grades))
    
    # If no clear signal, default to grade 4
    if max(grades) == 0:
        return 4
    
    return max_grade
