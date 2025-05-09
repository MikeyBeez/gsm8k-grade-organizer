import re

def classify_by_heuristics(problem):
    """
    Estimate grade level based on simple heuristics
    """
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
    
    # Assess numbers used in the problem
    numbers = re.findall(r'\d+', problem)
    if numbers:
        max_num = max([int(num) for num in numbers])
        if max_num <= 10:
            grades[0] += 1
        elif max_num <= 100:
            grades[1] += 1
        elif max_num <= 1000:
            grades[2] += 1
        else:
            grades[3] += 1
    
    # Determine the most likely grade
    max_grade = grades.index(max(grades))
    
    # If no clear signal, default to grade 4
    if max(grades) == 0:
        return 4
    
    return max_grade
