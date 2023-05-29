import math
import time


class Matrix:
    def __init__(self, size):
        self.size = size
        self.data = [[0] * size for _ in range(size)]

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])

    def __add__(self, other):
        if isinstance(other, Matrix) and self.size == other.size:
            result = Matrix(self.size)
            for i in range(self.size):
                for j in range(self.size):
                    result[i][j] = self[i][j] + other[i][j]
            return result
        else:
            raise ValueError("Matrices must have the same dimensions for addition.")

    def __mul__(self, other):
        if isinstance(other, Matrix) and self.size == other.size:
            result = Matrix(self.size)
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        result[i][j] += self[i][k] * other[k][j]
            return result
        else:
            raise ValueError("Matrices must have the same dimensions for multiplication.")

    def __sub__(self, other):
        if isinstance(other, Matrix) and self.size == other.size:
            result = Matrix(self.size)
            for i in range(self.size):
                for j in range(self.size):
                    result[i][j] = self[i][j] - other[i][j]
            return result
        else:
            raise ValueError("Matrices must have the same dimensions for subtraction.")

    def initialize_std_mat(self, e=4):

        self.data[0][0] = 5 + e
        self.data[0][1] = -1
        self.data[0][2] = -1

        self.data[1][0] = -1
        self.data[1][1] = 5 + e
        self.data[1][2] = -1
        self.data[1][3] = -1

        self.data[self.size - 2][self.size - 4] = -1
        self.data[self.size - 2][self.size - 3] = -1
        self.data[self.size - 2][self.size - 2] = 5 + e
        self.data[self.size - 2][self.size - 1] = -1

        self.data[self.size - 1][self.size - 3] = -1
        self.data[self.size - 1][self.size - 2] = -1
        self.data[self.size - 1][self.size - 1] = 5 + e
        for i in range(2, self.size - 2):
            self.data[i][i] = 5 + e
            self.data[i][i + 1] = -1
            self.data[i][i + 2] = -1
            self.data[i][i - 1] = -1
            self.data[i][i - 2] = -1

    def initialize_std_vec(self):
        f = 9
        for i in range(self.size):
            self.data[i][0] = math.sin(i * (f + 1))

    def copy(self):
        copy_matrix = Matrix(self.size)
        for i in range(self.size):
            copy_matrix[i] = self[i].copy()
        return copy_matrix

    def norm(self):
        sum_of_squares = 0
        for i in range(self.size):
            for j in range(self.size):
                sum_of_squares += self[i][j] ** 2
        return math.sqrt(sum_of_squares)

    def normalizacja(self, r, b):
        return (self * r - b).norm()

    def jacobi_solve(self, b, max_iterations=25, tolerance=1e-9):
        if not isinstance(b, Matrix) or self.size != b.size:
            raise ValueError("Input matrix dimensions must match.")

        x = Matrix(self.size)  # Initialize the solution vector
        x_prev = Matrix(self.size)  # Initialize the previous solution vector
        iteration_ctr = 0
        res_norm = self.norm()
        res = [res_norm]

        start = time.time()
        while res_norm > tolerance:
            if iteration_ctr > max_iterations:
                end = time.time()
                return iteration_ctr, end - start, res_norm, res
            # for iteration in range(max_iterations):
            # Update the solution vector
            for i in range(self.size):
                sum_val = 0
                for j in range(self.size):
                    if i != j:
                        sum_val += self[i][j] * x_prev[j][0]
                x[i][0] = (b[i][0] - sum_val) / self[i][i]

            iteration_ctr = iteration_ctr + 1
            res_norm = (x - x_prev).norm()
            res.append(res_norm)
            # Check for convergence
            if (x - x_prev).norm() < tolerance:
                break

            # Update the previous solution vector
            x_prev = x.copy()

        end = time.time()
        return iteration_ctr, end - start, res_norm, res

    def gauss_solve(self, b, max_iterations=25, tolerance=1e-9):
        if not isinstance(b, Matrix) or self.size != b.size:
            raise ValueError("Input matrix dimensions must match.")

        x = Matrix(self.size)  # Initialize the solution vector
        x_prev = Matrix(self.size)  # Initialize the previous solution vector
        iteration_ctr = 0
        res_norm = self.norm()
        res = [res_norm]

        start = time.time()
        while res_norm > tolerance:
            if iteration_ctr > max_iterations:
                end = time.time()
                return iteration_ctr, end - start, res_norm, res
            # for iteration in range(max_iterations):
            # Update the solution vector
            for i in range(self.size):
                sum_val = 0
                for j in range(i):
                    sum_val += self[i][j] * x[j][0]
                for j in range(i + 1, self.size):
                    sum_val += self[i][j] * x_prev[j][0]

                x[i][0] = (b[i][0] - sum_val) / self[i][i]

            iteration_ctr = iteration_ctr + 1
            res_norm = (x - x_prev).norm()
            res.append(res_norm)
            # Check for convergence
            if (x - x_prev).norm() < tolerance:
                break

            # Update the previous solution vector
            x_prev = x.copy()

        end = time.time()
        return iteration_ctr, end - start, res_norm, res

    def lu_factorization(self):
        if self.size != len(self[0]):
            raise ValueError("LU factorization is only applicable to square matrices.")

        L = Matrix(self.size)
        U = Matrix(self.size)

        for i in range(self.size):
            # Compute U matrix
            for j in range(i, self.size):
                sum_val = 0
                for k in range(i):
                    sum_val += L[i][k] * U[k][j]
                U[i][j] = self[i][j] - sum_val

            # Compute L matrix
            for j in range(i + 1, self.size):
                sum_val = 0
                for k in range(i):
                    sum_val += L[j][k] * U[k][i]
                L[j][i] = (self[j][i] - sum_val) / U[i][i]

            # Set diagonal elements of L to 1
            L[i][i] = 1

        return L, U

    def lu_solve(self, b):
        if self.size != len(self[0]):
            raise ValueError("LU factorization is only applicable to square matrices.")
        if not isinstance(b, Matrix) or self.size != b.size:
            raise ValueError("Input matrix dimensions must match.")

        start = time.time()
        L, U = self.lu_factorization()

        y = Matrix(self.size)  # Initialize intermediate variable y (Ly = b)
        x = Matrix(self.size)  # Initialize solution variable x (Ux = y)

        # Solve Ly = b for y using forward substitution
        for i in range(self.size):
            sum_val = 0
            for j in range(i):
                sum_val += L[i][j] * y[j][0]
            y[i][0] = (b[i][0] - sum_val) / L[i][i]

        # Solve Ux = y for x using backward substitution
        for i in reversed(range(self.size)):
            sum_val = 0
            for j in range(i + 1, self.size):
                sum_val += U[i][j] * x[j][0]
            x[i][0] = (y[i][0] - sum_val) / U[i][i]

        end = time.time()

        norm = self.normalizacja(x, b)

        return end - start, norm
