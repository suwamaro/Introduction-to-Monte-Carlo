import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 8  # Lattice size
N = L**2  # Number of spins
T_c = 2 / np.log(1+np.sqrt(2))  # Critical temperature = 2.269185314213...
steps = 2**13  # Number of Monte Carlo steps
n_simulations = 10  # Number of independent Markov chains
burn_in = steps // 2  # Thermalization step

def delta_E(spins, x, y):
    # Calculate the energy change for flipping a spin at (x, y).
    spin = spins[x, y]
    neighbors = spins[(x-1)%L, y] + spins[(x+1)%L, y] + \
                spins[x, (y-1)%L] + spins[x, (y+1)%L]
    return 2 * spin * neighbors

def Metropolis(spins, T):
    # Perform a sequential sweep of Metropolis updates.
    for x in range(L):
        for y in range(L):
            dE = delta_E(spins, x, y)
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                spins[x, y] *= -1

def magnetization(spins):
    # Calculate the magnetization density.
    return np.sum(spins) / N

def binning_analysis(data):
    # Binning analysis
    nsamples = len(data)
    bin_sizes = 2**np.arange(int(np.log2(nsamples/2)))
    vars = []
    data_array = np.array(data)
    for bin_size in bin_sizes:
        binned_data = np.mean(data_array[:nsamples//bin_size*bin_size].reshape(-1, bin_size), axis=1)
        vars.append(np.var(binned_data) * bin_size / nsamples)
    return bin_sizes, vars
    
variances = []
for sim in range(n_simulations):
    spins = np.random.choice([-1, 1], size=(L, L))  # Initial state

    # Observable
    mag2 = []

    # Main simulation
    for step in range(steps):
        Metropolis(spins, T_c)
        if step >= burn_in:
            mag = magnetization(spins)
            mag2.append(mag**2)

    # Binning
    bin_sizes, vars = binning_analysis(mag2)
    variances.append(vars)

# Average variances
variances = np.array(variances).reshape(n_simulations, -1)
var_ave = np.mean(variances, axis=0)
var_std = np.std(variances, axis=0, ddof=1)

# Plot results
plt.title('Number of samples = '+str(steps - burn_in)+',  Number of simulations = '+str(n_simulations))
plt.xlabel('Bin size')
plt.ylabel(r'$(\sigma^{(n)})^2 / (\sigma^{(1)})^2$')
plt.xscale('log')
plt.errorbar(bin_sizes, var_ave/var_ave[0], yerr=var_std/var_ave[0], fmt='o-', color='r', label='Average')
n_plots = 2
for plot_i in range(n_plots):
    plt.plot(bin_sizes, variances[plot_i]/variances[plot_i,0], 'x-', label='Simulation '+str(plot_i))
plt.legend(loc=2)
plt.show()