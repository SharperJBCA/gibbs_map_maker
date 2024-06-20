from gibbs_map_maker.matrix import Matrix
from gibbs_map_maker.vector import Vector
from gibbs_map_maker.gibbs_sampler import GibbsSampler
from gibbs_map_maker.convergence import check_convergence

# Load or generate the input data
P = Matrix(...)  # Load or generate matrix P
F = Matrix(...)  # Load or generate matrix F
G = Matrix(...)  # Load or generate matrix G
H = Matrix(...)  # Load or generate matrix H (additional matrix)
d = Vector(...)  # Load or generate vector d

# Initialize the parameter vectors
m = Vector(...)  # Initialize m to zero or random values
a = Vector(...)  # Initialize a to zero or random values
alpha = Vector(...)  # Initialize α to zero or random values
beta = Vector(...)  # Initialize β to zero or random values (additional parameter)

# Create an instance of the GibbsSampler class
sampler = GibbsSampler(matrices=[P, F, G, H], vector=d)
sampler.set_initial_values(m, a, alpha, beta)
sampler.set_convergence_criteria(tolerance=1e-6, max_iterations=1000)

# Run the Gibbs sampling iterations
sampler.run()

# Retrieve and print the final solutions
m_solution = sampler.get_solution('m')
a_solution = sampler.get_solution('a')
alpha_solution = sampler.get_solution('alpha')
beta_solution = sampler.get_solution('beta')

print("Solution for m:", m_solution)
print("Solution for a:", a_solution)
print("Solution for α:", alpha_solution)
print("Solution for β:", beta_solution)

# Plot the convergence history
residual_norms = sampler.get_residual_norms()
iterations = range(1, len(residual_norms) + 1)
plt.plot(iterations, residual_norms, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Convergence History')
plt.yscale('log')
plt.grid(True)
plt.show()