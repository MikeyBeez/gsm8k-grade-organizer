# Curriculum Learning in Mathematical Reasoning: Effects of Training Order on Model Performance

## Abstract

We investigate the effects of curriculum learning in the context of mathematical reasoning tasks. Using the Grade School Math 8K (GSM8K) dataset, we explore whether training models on a curriculum ordered by grade level (K-8) leads to better performance compared to the standard random ordering of training examples. Our experiments with DistilBERT on a grade-classified dataset show that curriculum ordering yields significant advantages for foundational mathematical concepts (97.43% error reduction in baseline experiments) and leads to specialized learning patterns. While the curriculum-ordered model achieved 10% accuracy on kindergarten-level problems after extended training, the randomly ordered model developed broader but shallower competence across grade levels. These findings provide empirical evidence that training order significantly impacts model specialization, suggesting that curriculum learning strategies should be tailored to specific educational objectives: deeper mastery of fundamentals or broader coverage across difficulty levels. Notably, all experiments were conducted on consumer-grade hardware (Apple Mac Mini), demonstrating that meaningful machine learning research can be performed without specialized equipment.

## 1. Introduction

Curriculum learning, inspired by human educational practices, proposes that models should learn concepts in a meaningful sequence rather than through random exposure to training examples. This approach suggests starting with simpler concepts before progressing to more complex ones, mirroring how students advance through educational systems. While curriculum learning has shown promise in various domains, its efficacy and specific effects in mathematical reasoning remain underexplored.

Mathematical reasoning presents a particularly suitable domain for curriculum learning investigation due to its inherently hierarchical nature. Concepts taught in higher grades typically build upon foundations established in earlier grades. The GSM8K dataset, comprising grade school math word problems, offers an excellent opportunity to test curriculum learning hypotheses by organizing training examples according to grade-level difficulty.

This paper addresses two key research questions:
1. Does training on a curriculum ordered by grade level improve model performance on mathematical reasoning tasks compared to standard random ordering?
2. How does the ordering of training examples affect the specialization patterns and error distributions across different grade levels?

## 2. Methodology

### 2.1 Dataset Organization

We utilized the GSM8K dataset, which contains 7,473 diverse grade school math word problems. To implement curriculum learning, we first needed to classify each problem by grade level (K-8). We developed a heuristic classifier that analyzed problem text for grade-specific indicators, including:

- Mathematical concepts present (e.g., counting, multiplication, fractions)
- Vocabulary complexity
- Problem length
- Numerical magnitude
- Operation types

The resulting grade distribution showed a concentration of problems in the lower grades:
- Kindergarten (K): 3,386 problems (45.3%)
- Grade 1: 2,192 problems (29.3%)
- Grade 2: 1,257 problems (16.8%)
- Grade 3: 253 problems (3.4%)
- Grade 4: 4 problems (0.1%)
- Grade 5: 0 problems (0%)
- Grade 6: 380 problems (5.1%)
- Grade 7: 0 problems (0%)
- Grade 8: 1 problem (0%)

One of the key contributions of our work is the creation of this grade-organized version of the GSM8K dataset, which we make available to the research community. The grade-classified dataset can be generated using our code repository, allowing researchers to experiment with different curriculum learning strategies beyond the sequential approach we explored.

### 2.2 Experimental Setup

We conducted two sets of experiments:

**Baseline Experiment:**
- 200 training examples, 5 epochs, learning rate 1e-5
- Test sample: 50 examples, 20 examples per grade level for grade-specific evaluation

**Extended Experiment:**
- 1,000 training examples, 10 epochs, learning rate 1e-5
- Test sample: 100 examples, 30 examples per grade level for grade-specific evaluation

For each experiment, we trained two models using DistilBERT as the base architecture:
1. **Original Model**: Trained on examples in their original random order
2. **Ordered Model**: Trained on examples ordered by grade level (K→8)

We evaluated the models on both overall performance and grade-specific performance metrics, including:
- Accuracy (proportion of problems within 5% of correct answer)
- Average error
- Average relative error

### 2.3 Computational Resources

All experiments were conducted on an Apple Mac Mini with M-series chip, demonstrating that meaningful machine learning research can be performed on consumer-grade hardware without access to specialized equipment or extensive GPU resources. The baseline experiment completed in approximately 10 minutes, while the extended experiment required about 60 minutes of computation time, making this research approach accessible to researchers with limited computational resources.

## 3. Results

### 3.1 Baseline Experiment

In the baseline experiment with 200 training examples, we observed:

**Overall Performance:**
- Original Model: 0% accuracy, average error of 57,808.63
- Ordered Model: 0% accuracy, average error of 1,487.26
- **Error Reduction**: 97.43% from curriculum ordering

**Grade-Level Performance:**
- Kindergarten: 0% (Original) vs. 5% accuracy (Ordered)
- Grade 1: 0% (Original) vs. 10% accuracy (Ordered)
- Grade 3: 0% (Original) vs. 5% accuracy (Ordered)
- Grade 6: 0% (Original) vs. 5% accuracy (Ordered)

The ordered model showed better performance across multiple grade levels, with particularly strong results on Grade 1 problems.

### 3.2 Extended Experiment

In the extended experiment with 1,000 training examples, we observed:

**Overall Performance:**
- Original Model: 4% accuracy, average error of 1,630.01
- Ordered Model: 0% accuracy on random test set, average error of 5,315.00

**Grade-Level Performance:**
- Kindergarten: 3.33% (Original) vs. 10% accuracy (Ordered)
- Grade 2: 3.33% (Original) vs. 0% accuracy (Ordered)
- Grade 4: 25% (Original) vs. 0% accuracy (Ordered)
- Grade 6: 3.33% (Original) vs. 0% accuracy (Ordered)
- Grade 8: 100% (Original) vs. 0% accuracy (Ordered) (single problem)

The results reveal intriguing specialization patterns: The extended original model showed broader but shallower competence across grade levels, while the ordered model focused its competence on kindergarten-level problems.

## 4. Discussion

### 4.1 Curriculum Learning Benefits

The baseline experiment demonstrated clear benefits from curriculum learning, with a remarkable 97.43% reduction in average error. The ordered model outperformed the original model across several grade levels, suggesting that curriculum learning helps establish stronger foundations for mathematical reasoning.

However, the extended experiment revealed a more nuanced picture. While the curriculum-ordered model doubled its kindergarten accuracy from the baseline (5% to 10%), it showed no improvement on other grades. In contrast, the randomly-ordered model developed broader competence across multiple grade levels.

### 4.2 Specialization Patterns

Our results indicate that training order significantly influences what models specialize in learning:

1. **Curriculum ordering promotes focused learning**: The ordered model achieved higher accuracy on foundational (kindergarten) problems but showed limited competence on higher grades.

2. **Random ordering promotes broader learning**: The extended original model developed competence across multiple grade levels, culminating in an impressive 25% accuracy on Grade 4 problems and solving the single Grade 8 problem.

These patterns suggest that curriculum learning helps models master foundational concepts but may create optimization challenges when transitioning to more complex problems. Random ordering, while less efficient initially, may foster more flexible learning that spans difficulty levels.

### 4.3 Error Distributions

Analysis of error magnitudes across grades revealed that both models struggled more with higher-grade problems, but in different ways. The original model showed high variance in error rates across grades, while the ordered model's errors were more consistent across early grades but increased substantially for higher grades.

This error pattern aligns with curriculum learning theory, which predicts stronger performance on the types of problems encountered early in training. The ordered model demonstrated lower errors on early-grade problems, suggesting it built stronger foundations at the expense of higher-grade competence.

## 5. Limitations

Several limitations affect the interpretation of our results:

1. **Dataset imbalance**: The GSM8K dataset has a heavy concentration of problems in grades K-2, which may bias model learning.

2. **Model architecture**: DistilBERT is not specifically designed for mathematical reasoning, potentially limiting achievable performance.

3. **Grade classification accuracy**: Our heuristic classifier may have introduced biases in problem categorization.

4. **Training sample size**: Even our extended experiment used only a fraction of the available training data.

5. **Evaluation metrics**: Accuracy within 5% may be too stringent for some mathematical tasks.

## 6. Conclusion

Our investigation into curriculum learning for mathematical reasoning yields several key insights. Training order significantly influences not just performance, but what models specialize in learning. Curriculum ordering by grade level produces substantial benefits for foundational mathematical concepts but appears to create specialized rather than generalized competence.

The choice between curriculum learning and random ordering depends on educational objectives: deeper mastery of fundamentals or broader coverage across difficulty levels. Future work should explore balanced curriculum approaches that combine depth and breadth advantages, specialized architectures for mathematical reasoning, and more sophisticated sequencing strategies that adapt to model progress.

These findings contribute to our understanding of how training dynamics shape model learning patterns and suggest that curriculum design remains a fertile area for improving mathematical reasoning capabilities in language models. Additionally, our successful execution of these experiments on consumer-grade hardware demonstrates the democratization of ML research, making it accessible to a broader community of researchers without requiring specialized computing infrastructure.

## Code and Data Availability

We have made all code and training scripts available in our public GitHub repository: [https://github.com/MikeyBeez/gsm8k-grade-organizer](https://github.com/MikeyBeez/gsm8k-grade-organizer)

The repository includes:
- Source code for the classification, training, and evaluation
- Scripts to reproduce all experiments
- Visualization code for analyzing results

Most importantly, our repository provides the tools to generate the grade-ordered GSM8K dataset that was central to our experiments. Researchers interested in curriculum learning can use our grade classification system to create ordered training sets for their own experiments. The ordered training set organizes problems by K-8 grade levels, providing a valuable resource for investigating different curriculum learning strategies in mathematical reasoning.

## Appendix: Grade Distribution Analysis

**Problem Distribution by Grade Level:**
- Kindergarten (K): 3,386 problems (45.3%)
- Grade 1: 2,192 problems (29.3%)
- Grade 2: 1,257 problems (16.8%)
- Grade 3: 253 problems (3.4%)
- Grade 4: 4 problems (0.1%)
- Grade 6: 380 problems (5.1%)
- Grade 8: 1 problem (0%)

**Average Problem Length by Grade:**
- Kindergarten: 44.0 words
- Grade 1: 46.7 words
- Grade 2: 48.1 words
- Grade 3: 47.8 words
- Grade 4: 26.2 words
- Grade 6: 34.1 words
- Grade 8: 47.0 words

**Average Maximum Number by Grade:**
- Kindergarten: 399.2
- Grade 1: 1,142.8
- Grade 2: 494.0
- Grade 3: 5,437.0
- Grade 4: 102.5
- Grade 6: 1,391.1
- Grade 8: 50.0
