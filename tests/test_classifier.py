import unittest
from src.grade_classifier import estimate_grade_level

class TestGradeClassifier(unittest.TestCase):
    def test_estimate_grade_level_prompt_generation(self):
        # Test that the prompt generator creates valid prompts
        problem = "Janet has 5 apples. She gives 2 to her friend. How many does she have left?"
        prompt = estimate_grade_level(problem)
        
        # Check that the prompt contains the problem
        self.assertIn(problem, prompt)
        
        # Check that the prompt contains grade level information
        self.assertIn("Kindergarten", prompt)
        self.assertIn("Grade 8", prompt)

if __name__ == '__main__':
    unittest.main()
