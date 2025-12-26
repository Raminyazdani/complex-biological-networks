"""
Symbolic Matrix Rank Calculator

Performs symbolic linear algebra operations using SymPy.
Computes matrix rank through symbolic row reduction.

Features:
- Symbolic matrix creation
- Row operations (swapping, row reduction)
- Symbolic rank calculation
- Matrix transformation tracking
"""

import sympy as sp


class SymbolicRankMatrix:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.Matrix = sp.Matrix(n, m, lambda i, j: sp.Symbol(f'a{i + 1}{j + 1}'))

    # Function to display the symbolic matrix
    def display_matrix(self):
        sp.pprint(self.Matrix)

    # Function to perform row swapping
    def swap(self, row1, row2):
        self.Matrix.row_swap(row1, row2)

    # Function to perform row operations
    def row_operation(self, row, target_row, multiplier):
        self.Matrix.row_op(target_row, lambda v, j: v - multiplier * self.Matrix[row, j])

    # Function to calculate the rank symbolically
    def rank_of_matrix(self):
        rank = self.m
        for row in range(rank):
            if self.Matrix[row, row] != 0:
                for col in range(self.n):
                    if col != row:
                        multiplier = self.Matrix[col, row] / self.Matrix[row, row]
                        self.row_operation(row, col, multiplier)
            else:
                reduce = True
                for i in range(row + 1, self.n):
                    if self.Matrix[i, row] != 0:
                        self.swap(row, i)
                        reduce = False
                        break
                if reduce:
                    rank -= 1
                    self.Matrix[:, row] = self.Matrix[:, rank]
                row -= 1

        return rank


# Driver Code
if __name__ == '__main__':
    n, m = 5, 3  # Dimensions of the matrix
    sym_matrix = SymbolicRankMatrix(n, m)
    print("Symbolic Matrix:")
    sym_matrix.display_matrix()

    rank = sym_matrix.rank_of_matrix()
    print(f"\nSymbolic Rank Calculation completed.")
    print(f"Rank of the symbolic matrix is: {rank}")
    print("\nTransformed Matrix:")
    sym_matrix.display_matrix()
