# CMPUT366-Project
Project Report: Sudoku Solver


Introduction
Sudoku, a popular logic-based puzzle, has inspired the development of various solving algorithms. This project focuses on implementing and evaluating a Sudoku solver with an emphasis on advanced techniques such as,the integration of Learning No-Goods and the Degree Heuristic for variable selection. The objective is to enhance the efficiency of Sudoku puzzle solving through comparative analysis of these techniques.

Methods
Learning No-Goods in Backtracking
The BacktrackingLearnNoGoods class extends the backtracking algorithm by learning and avoiding previously encountered conflicting states. This technique, known as Learning No-Goods, aims to prevent redundant exploration, significantly enhancing the solver's efficiency.
Initialization: Inherits from the backtracking class and introduces Learning No-Goods. The constructor initializes a set to store learned no-goods.
Search Function: Implements backtracking search with inference and Learning No-Goods. The search algorithm explores the solution space iteratively, incorporating learned no-goods to avoid revisiting conflicting states.


Degree Heuristic for Variable Selection
The DegreeHeuristic class introduces a variable selection strategy based on the degree of conflicts. Variables with higher degrees, reflecting more constraints, are prioritized during the search. This heuristic is designed to accelerate the solving process by focusing on the most constrained variables.
Variable Selection: Introduces a degree-based variable selection heuristic. It iterates through the Sudoku grid to identify variables with multiple legal values, prioritizing those with higher degrees.
Degree Calculation: Determines the degree of a variable based on conflicts in the same row, column, and unit. The degree reflects how constrained a variable is.
Conflict Check: Verifies if a value assignment leads to a conflict in the Sudoku grid. This ensures that the chosen variable assignment is consistent with the puzzle constraints.


Evaluation: MRV vs. FA vs. Degree Heuristic in Backtracking with Learning No-Goods
MRV (Minimum Remaining Values):
The MRV heuristic, when integrated with backtracking learning no-goods, prioritizes variables with the fewest remaining legal values. This approach aims to reduce the search space dynamically by leveraging learned information from past decisions, making it effective for puzzles with clearly defined constraints.

FA (First-Available) Heuristic:
In the context of backtracking learning no-goods, the FA heuristic continues to select variables based on their order, simplifying the search process. This method excels in scenarios where the order of variable selection is crucial. By incorporating learning from past failures, it enhances its ability to make informed choices.

Degree Heuristic:
The Degree Heuristic, when used with backtracking learning no-goods, considers the degree of conflicts and prioritizes variables with higher degrees. This strategy remains effective in efficiently exploring the solution space by focusing on the most constrained variables. The integration of learning no-goods enhances its capability to avoid repeating unsuccessful paths, contributing to more efficient search processes.

Experimentation
To assess the solver's performance, we conducted experiments on a diverse set of Sudoku puzzles, including easy, medium, and challenging ones. The metrics considered include the time taken to solve puzzles, the number of backtracks, and the efficiency of each heuristic in finding solutions.

Backtracking Results:

Degree Heuristic:
Demonstrates varying performance, with some instances of high running times. Extreme outliers of values ranging between 1500 - 2000+.
May struggle with specific Sudoku instances, leading to longer solving times.

First Available:
Inconsistent performance, with both fast and relatively slow solving times.
Shows variability in efficiency across different puzzles.

MRV (Minimum Remaining Values):
Generally fast and consistent performance.
Efficient in solving Sudoku puzzles, displaying reliable results.

Learning No Goods Backtracking Results:

Degree Heuristic:
Moderate running times, with occasional spikes.
Similar trends to the non-learning version but with potential improvements.

First Available:
Variable performance, with mixed efficiency.
Less consistent compared to MRV.

MRV (Minimum Remaining Values):
Generally fast and consistent solving times.
Maintains efficiency and reliability with the learning aspect.

Overall Comparison:
Backtracking vs. Learning No Goods Backtracking:
Learning No Goods Backtracking introduces improvements in certain scenarios.
MRV consistently outperforms Degree Heuristic and First Available in both versions.

Recommendations:
Choose MRV for Learning No Goods Backtracking: MRV continues to demonstrate efficiency and consistency, making it a suitable choice.
Evaluate Degree Heuristic and First Available: Further analysis and optimization may be beneficial, particularly for Degree Heuristic and First Available, to enhance their performance in specific puzzle scenarios.
There is no clear trend across all three heuristics, indicating that the performance depends on the specific characteristics of each puzzle.

Conclusion
In conclusion, this project explores and compares the efficacy of MRV, FA, and Degree Heuristic in the context of Sudoku puzzle solving using the Learning No Goods Algorithm. The results and analysis presented in the subsequent sections aim to provide a comprehensive understanding of the strengths and weaknesses of each approach, contributing valuable insights to the field of puzzle-solving algorithms.

References
Russell, S., & Norvig, P. (2010). Artificial Intelligence: A Modern Approach. Pearson.
Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.
Simonis, H. (2005). On the definition of constraint propagation. In Principles and Practice of Constraint Programming (pp. 4-20). Springer.
