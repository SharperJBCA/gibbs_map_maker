import numpy as np
from tqdm import tqdm 

class GibbsSampler:
    def __init__(self, matrices, vector_d, initial_values=None, tolerance=1e-6, max_iterations=500):
        self.matrices = matrices
        self.vector_d = vector_d
        self.initial_values = initial_values if initial_values is not None else {}
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.parameter_vectors = {}
        self.parameter_chains = {}
        self.initialize_parameter_vectors()

    def initialize_parameter_vectors(self):
        for matrix in self.matrices:
            name = matrix.__class__.__name__.lower()
            self.parameter_vectors[name] = self.initial_values.get(name, np.zeros(matrix.cols))

    def gibbs_sampling(self):
        for iteration in tqdm(range(self.max_iterations),desc='Gibbs Sampling'):
            for matrix in self.matrices:
                name = matrix.__class__.__name__.lower()
                if not name in self.parameter_chains:
                    self.parameter_chains[name] = []
                rhs = self.compute_rhs(matrix, name)
                self.parameter_vectors[name] = self.conjugate_gradient(matrix, rhs)
                self.parameter_chains[name].append(self.parameter_vectors[name])
            if self.check_convergence():
                break

    def compute_rhs(self, matrix, name):
        rhs = self.vector_d.copy()
        for other_matrix in self.matrices:
            if other_matrix != matrix:
                other_name = other_matrix.__class__.__name__.lower()
                rhs -= other_matrix.forward(self.parameter_vectors[other_name])
        for this_matrix in self.matrices:
            if this_matrix == matrix:
                name = this_matrix.__class__.__name__.lower()
                rhs = this_matrix.backward(rhs)

        return rhs

    def conjugate_gradient(self, matrix, rhs, tolerance=1e-6, max_iterations=1000):
        x = np.zeros(matrix.cols)
        r = rhs - matrix.bkwdfwd(x)
        p = r.copy()
        for iteration in range(max_iterations):
            alpha = np.dot(r, r) / np.dot(p, matrix.bkwdfwd(p))
            if np.isnan(alpha):
                break
            x += alpha * p
            r_next = r - alpha * matrix.bkwdfwd(p)
            if np.linalg.norm(r_next) < tolerance:
                break
            beta = np.dot(r_next, r_next) / np.dot(r, r)
            p = r_next + beta * p
            r = r_next
        return x

    def check_convergence(self):
        # Check for convergence based on tolerance and maximum iterations
        # Implement your convergence criteria here
        pass

    def get_parameter_vectors(self):
        return self.parameter_vectors

    def set_initial_values(self, initial_values):
        self.initial_values = initial_values
        self.initialize_parameter_vectors()

    def set_convergence_criteria(self, tolerance, max_iterations):
        self.tolerance = tolerance
        self.max_iterations = max_iterations