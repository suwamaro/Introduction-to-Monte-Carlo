import random
import numpy as np
from scipy.special import gamma

def generate_random_vector(dimensions):
    return [random.uniform(-1, 1) for _ in range(dimensions)]

def estimate_pi(dim, num_samples):
    points_inside_circle = 0
    for _ in range(num_samples):
        x_vec = generate_random_vector(dim)
        norm = sum(x**2 for x in x_vec)
        if norm <= 1:
            points_inside_circle += 1
    return np.power(gamma(dim/2+1) * np.power(2, dim) * points_inside_circle / num_samples, 2/dim)

# Estimate pi
num_samples = 10 ** 6
print(f'Number of samples: {num_samples}')
dims = [2,4,6,8,10,12,14,16,18,20,22]
for d in dims:
    pi_estimate = estimate_pi(d, num_samples)
    error = abs(np.pi - pi_estimate)
    print(f'Dimension: {d}, Estimated Pi: {pi_estimate}, Absolute Error: {error}')
