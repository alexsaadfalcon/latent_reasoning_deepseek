# Textbook datasets

This directory contains problems from two math textbooks:
- Convex Optimization by Boyd and Vandenberghe (graduate level)
- Aspects of Combinatorics by Victor Bryant (undergraduate level)

There is one json file each for Chapter 2 of Convex Optimization and Chapter 1 of Aspects of Combinatorics. Each json file contains a list of problems, where each problem is represented as a dictionary with the following keys:
- `exericse`: Problem number in the textbook
- `text`: Problem text
- `type`: Type of problem, one of `answer-based`, `proof-based`, and `open-ended`
- `answer`: Answer, if the problem is `answer-based`
