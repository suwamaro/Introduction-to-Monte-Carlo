import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
# from IPython.display import HTML

# Parameters
sigma1, sigma2 = 1.0, 10.0
std_dev = np.array([0.5, 0.5])
steps = 2 ** 15
n_bins = 8
initial_sample = np.array([0.0, 0.0])

def target_density(x):
    # Target density: a bivariate normal distribution.
    # normalization = 2 * np.pi * sigma1 * sigma2
    return np.exp(-(x[0] - x[1])**2 / (2*sigma1**2) - (x[0] + x[1])**2 / (2*sigma2**2))

def MetropolisHastings(target_density, initial_sample, steps, n_bins, std_dev, obs1, obs2):
    samples = np.zeros((steps, 2))
    samples[0] = initial_sample
    burn_in = steps // 2
    bin_size = max((steps - burn_in) // n_bins, 1)
    o1 = 0
    o2 = 0
    n_samp = 0
    for i in range(1, steps):
        current_sample = samples[i-1]
        candidate_sample = np.random.normal(loc=current_sample, scale=std_dev)
        acceptance_prob = target_density(candidate_sample) / target_density(current_sample)
        if np.random.rand() < acceptance_prob:
            samples[i] = candidate_sample
        else:
            samples[i] = current_sample

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
samples = MetropolisHastings(target_density=target_density, initial_sample=initial_sample, steps=steps, n_bins=n_bins, std_dev=std_dev, obs1=obs1, obs2=obs2)

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
ax.set_title('Metropolis-Hastings algorithm')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
line, = ax.plot([], [], lw=1)
scatter = ax.scatter([], [], s=1)

# Initialization for the animation
def init():
    line.set_data([], [])
    scatter.set_offsets(np.empty((0, 2)))
    ax.grid(True)
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

# Create the animation
ani = animation.FuncAnimation(fig, update_line, init_func=init, frames=steps, fargs=(samples, line, scatter), interval=20, blit=True)
# ani.save('animation.gif', writer='imagemagick', fps=60)
plt.show()
# HTML(ani.to_html5_video())