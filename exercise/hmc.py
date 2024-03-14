import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

# Parameters
nuts = True
sigma1, sigma2 = 1.0, 10.0
beta = 1.0  # Inverse temperature
steps = 2 ** 15
n_bins = 8
initial_sample = np.array([0.0, 0.0])
step_size = 0.4
Delta = 1000
num_leapfrog_steps = 50  # If not nuts

def log_target_density(x, beta):
    # Target density: a bivariate normal distribution.
    E = (x[0] - x[1])**2 / (2*sigma1**2) + (x[0] + x[1])**2 / (2*sigma2**2)
    return - beta * E

def gradient_log_target_density(x, beta):
    grad_x1 = (x[0] - x[1]) / (sigma1**2) + (x[0] + x[1]) / (sigma2**2)
    grad_x2 = - (x[0] - x[1]) / (sigma1**2) + (x[0] + x[1]) / (sigma2**2)
    return - np.array([grad_x1, grad_x2]) * beta

def leapfrog(x, p, grad, beta, step_size, num_steps):
    x_new = np.copy(x)
    p_new = np.copy(p)
    
    p_new -= 0.5 * step_size * grad(x_new, beta)
    for _ in range(num_steps - 1):
        x_new += step_size * p_new
        p_new -= step_size * grad(x_new, beta)
    x_new += step_size * p_new
    p_new -= 0.5 * step_size * grad(x_new, beta)
    
    return x_new, p_new

def build_tree(x, p, u, v, j, dt, beta):
    # Recursively doubling the size of a binary tree
    if j == 0:
        x2, p2 = leapfrog(x, p, grad=gradient_log_target_density, beta=beta, step_size=v*dt, num_steps=1)
        H2 = -log_target_density(x2, beta) + 0.5 * np.sum(p2**2)
        if np.log(u) <= - H2:
            # Candidate
            C2 = [x2.tolist()]
        else:
            C2 = []
        
        if np.log(u) > Delta - H2:            
            # Too large energy
            s2 = False
        else:
            s2 = True

        return x2, p2, x2, p2, C2, s2
    else:
        xm, pm, xp, pp, C2, s2 = build_tree(x, p, u, v, j-1, dt, beta)
        if v == -1:
            xm, pm, _, _, C3, s3 = build_tree(xm, pm, u, v, j-1, dt, beta)
        else:
            _, _, xp, pp, C3, s3 = build_tree(xp, pp, u, v, j-1, dt, beta)

        s_uturn = np.dot(xp - xm, pm) >= 0 and np.dot(xp - xm, pp) >= 0
        s2 = s2 and s3 and s_uturn
        C2 += C3        
        return xm, pm, xp, pp, C2, s2

def HMC(target_log_density, gradient_log_density, beta, initial_sample, steps, step_size, num_leapfrog_steps, obs1, obs2, nuts=True):
    samples = np.zeros((steps, 2))
    samples[0] = initial_sample
    burn_in = steps // 2
    bin_size = max((steps - burn_in) // n_bins, 1)
    o1 = 0
    o2 = 0
    n_samp = 0

    for i in range(1, steps):
        x0 = samples[i-1]
        p0 = np.random.normal(size=2) # Initialize momentum
        if nuts:
            # No-U-turn sampler
            H = -target_log_density(x0, beta) + 0.5 * np.sum(p0**2)
            # Slice sampling
            u = np.random.rand() * np.exp(-H)
            xm, xp = x0, x0
            pm, pp = p0, p0
            j = 0
            C = [x0.tolist()]            
            s = True
            while s:
                v = np.random.choice([-1,1])
                if v == -1:
                    xm, pm, _, _, C2, s2 = build_tree(xm, pm, u, v, j, step_size, beta)
                else:
                    _, _, xp, pp, C2, s2 = build_tree(xp, pp, u, v, j, step_size, beta)
                
                if s2:
                    C += C2
                s_uturn = np.dot(xp - xm, pm) >= 0 and np.dot(xp - xm, pp) >= 0
                s = s2 and s_uturn
                j += 1

            # Uniform random choice
            samples[i] = random.choice(C)
        else:
            # Naive version
            proposed_x, proposed_p = leapfrog(x=x0, p=p0, grad=gradient_log_density, beta=beta, step_size=step_size, num_steps=num_leapfrog_steps)
        
            # Compute Hamiltonian for the current and proposed state
            current_H = -target_log_density(x0, beta) + 0.5 * np.sum(p0**2)
            proposed_H = -target_log_density(proposed_x, beta) + 0.5 * np.sum(proposed_p**2)
            
            # Acceptance criterion
            if np.random.rand() < np.exp((current_H - proposed_H)):
                samples[i] = proposed_x
            else:
                samples[i] = x0

        # Measurement
        if i >= burn_in:        
            _sum = sum(samples[i])
            o1 += _sum
            o2 += _sum ** 2
            n_samp += 1
            if n_samp == bin_size:
                obs1.append(o1/bin_size)
                obs2.append(o2/bin_size)
                o1 = 0
                o2 = 0
                n_samp = 0

    return samples

# Sample
obs1, obs2 = [], []
samples = HMC(log_target_density, gradient_log_target_density, beta, initial_sample, steps, step_size, num_leapfrog_steps, obs1, obs2, nuts=nuts)

# Observables
mu1, std1 = np.mean(obs1), np.std(obs1, ddof=1) / np.sqrt(len(obs1))
mu2, std2 = np.mean(obs2), np.std(obs2, ddof=1) / np.sqrt(len(obs2))
print(f'(x_1 + x_2): Sample mean: {mu1}, Sample standard deviation: {std1}')
print(f'(x_1 + x_2)^2: Sample mean: {mu2}, Sample standard deviation: {std2}')

# Set up the figure for animation
fig, ax = plt.subplots()
xlim = 1.5 * sigma2
ax.set_xlim([-xlim, xlim])
ax.set_ylim([-xlim, xlim])
ax.set_title('Hamiltonian Monte Carlo')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
line, = ax.plot([], [], lw=1)
scatter = ax.scatter([], [], s=1)

# Initialization for the animation
def init():
    line.set_data([], [])
    scatter.set_offsets(np.empty((0, 2)))
    return line, scatter,

# Update function for the animation that includes a line connecting the plots
def update_line(i, samples, line, scatter):
    line.set_data(samples[:i+1, 0], samples[:i+1, 1])
    scatter.set_offsets(samples[:i+1])
    return line, scatter,

# Two-sigma line
ellipse = Ellipse(xy=(0, 0), width=2*np.sqrt(2)*sigma2, height=2*np.sqrt(2)*sigma1, angle=45, edgecolor='k', fc='None', lw=2)
ax.add_patch(ellipse)
# Setting the aspect ratio to 'equal' so the ellipse isn't distorted
ax.set_aspect('equal')

# Display the plot
plt.grid(True)
# Create the animation
ani = animation.FuncAnimation(fig, update_line, init_func=init, frames=steps, fargs=(samples, line, scatter), interval=20, blit=True)
# ani.save('animation.gif', writer='imagemagick', fps=60)
plt.show()
