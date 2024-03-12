import sys
import numpy as np
import scipy.sparse.csgraph as csg
import matplotlib.pyplot as plt
sys.path.append('.')
from bootstrapping import bootstrapping

class IsingSwendsenWang:
    def __init__(self, L):
        self.L = L  # Lattice size
        self.N = L*L  # Number of spins
        self.spins = np.random.choice([-1, 1], size=(L, L))
        self.J = 1  # Interaction strength
        self.q1 = [2 * np.pi / self.L, 0]  # Smallest nonzero wave vector
        self.n_boot = 500  # Number of bootstrap samples
        self.T = 1.0

    def _bond(self, s1, s2):
        # Determine if the bond between spins s1 and s2 is active.
        return np.random.rand() < (1 - np.exp(-2 * self.J / self.T)) if s1 == s2 else 0

    def swendsen_wang_step(self):
        # Perform one step of the Swendsen-Wang algorithm.
        # Create a graph where nodes represent spins and edges represent bonds
        graph = np.zeros((self.N, self.N), dtype=int)
        for i in range(self.L):
            for j in range(self.L):
                # Connect bonds
                if self._bond(self.spins[i, j], self.spins[(i+1)%self.L, j]):
                    graph[i*self.L+j, ((i+1)%self.L)*self.L+j] = 1
                if self._bond(self.spins[i, j], self.spins[i, (j+1)%self.L]):
                    graph[i*self.L+j, i*self.L+(j+1)%self.L] = 1 

        # Identify clusters using connected components
        _, labels = csg.connected_components(graph, directed=False)

        # Flip each cluster with 50% probability
        for cluster in np.unique(labels):
            if np.random.rand() < 0.5:
                self.spins[labels.reshape(self.L, self.L) == cluster] *= -1

    def simulate(self, steps, n_bins):
        # Simulate the Ising model using the Swendsen-Wang algorithm.
        ms, m2s, m4s, m_abss, es, e2s, sfs, sf1s = [], [], [], [], [], [], [], []
        m, m2, m4, m_abs, e, e2, sf, sf1 = 0, 0, 0, 0, 0, 0, 0, 0        
        bin_size = steps // 2 // n_bins
        n_samp = 0
        for t in range(steps):
            # Update
            self.swendsen_wang_step()
            if t >= steps // 2:
                # Measurements
                _e = self.calculate_energy()
                _m = self.calculate_magnetization()
                _sf1 = self.calculate_structure_factor(self.q1)
                m += _m
                m2 += _m ** 2
                m4 += _m ** 4
                m_abs += np.abs(_m)
                e += _e
                e2 += _e ** 2
                sf += _m ** 2 * self.N
                sf1 += _sf1
                n_samp += 1
                if n_samp == bin_size:
                    ms.append(m/bin_size)
                    m2s.append(m2/bin_size)
                    m4s.append(m4/bin_size)
                    m_abss.append(m_abs/bin_size)
                    es.append(e/bin_size)
                    e2s.append(e2/bin_size)
                    sfs.append(sf/bin_size)
                    sf1s.append(sf1/bin_size)
                    m, m2, m4, m_abs, e, e2, sf, sf1 = 0, 0, 0, 0, 0, 0, 0, 0        
                    n_samp = 0

        ms = np.array(ms)
        m2s = np.array(m2s)
        m4s = np.array(m4s)
        m_abss = np.array(m_abss)
        es = np.array(es)
        e2s = np.array(e2s)
        sfs = np.array(sfs)
        sf1s = np.array(sf1s)
        return ms, m2s, m4s, m_abss, es, e2s, sfs, sf1s

    def average(self, bins):
        mu, sigma = np.mean(bins), np.std(bins, ddof=1) / np.sqrt(len(bins))
        return mu, sigma

    def simulate_temperature_range(self, Tmin, Tmax, n_Ts, steps_per_temp, n_bins):
        # Simulate over a range of temperatures.
        temperatures = np.linspace(Tmin, Tmax, n_Ts)
        energies = []
        specific_heats = []
        susceptibilities = []
        Binder_cumulants = []
        correlation_length_Ls = []
        
        for T in temperatures:
            self.T = T
            self.spins = np.random.choice([-1, 1], size=(self.L, self.L))  # Reinitialize spins
            ms, m2s, m4s, m_abss, es, e2s, sfs, sf1s = self.simulate(steps_per_temp, n_bins)
            energy, energy_err = self.average(es)
            sheat, sheat_err = self.calculate_specific_heat(es, e2s, T)
            sus, sus_err = self.calculate_susceptibility(m2s, m_abss, T)
            binder, binder_err = self.calculate_binder(m2s, m4s)
            corr_len_L, corr_len_L_err = self.calculate_correlation_length_L(sfs, sf1s)

            energies.append([T, energy, energy_err])
            specific_heats.append([T, sheat, sheat_err])
            susceptibilities.append([T, sus, sus_err])
            Binder_cumulants.append([T, binder, binder_err])
            correlation_length_Ls.append([T, corr_len_L, corr_len_L_err])

        energies = np.array(energies) 
        specific_heats = np.array(specific_heats)
        susceptibilities = np.array(susceptibilities)
        Binder_cumulants = np.array(Binder_cumulants)
        correlation_length_Ls = np.array(correlation_length_Ls)
        return energies, specific_heats, susceptibilities, Binder_cumulants, correlation_length_Ls

    # Include other methods (swendsen_wang_step, simulate, calculate_energy, calculate_magnetization) here...

    def calculate_energy(self):
        # Calculate the energy of the current state.
        energy = 0
        for i in range(self.L):
            for j in range(self.L):
                S = self.spins[i, j]
                neighbors = self.spins[(i+1)%self.L, j] + self.spins[i, (j+1)%self.L] + self.spins[(i-1)%self.L, j] + self.spins[i, (j-1)%self.L]
                energy -= self.J * S * neighbors
        return energy / 2.0  # Each bond counted twice

    def calculate_magnetization(self):
        # Calculate the magnetization of the current state.
        return np.sum(self.spins) / self.N

    def calculate_structure_factor(self, q):
        # Calculate the structure factors
        Sq = 0
        for i in range(self.L):
            for j in range(self.L):
                S = self.spins[i, j]
                Sq += S * np.exp(1j * (i * q[0] + j * q[1]))
        sf = (Sq * np.conj(Sq)).real / self.N
        return sf

    def f_var(self, x, x2):
        return x2 - x * x

    def f_binder(self, x, y):
        return x * x / y

    def f_xi_L(self, x, y):
        return np.sqrt(x / y - 1)

    def calculate_specific_heat(self, e, e2, T):
        mu, sigma = bootstrapping(e, e2, self.n_boot, self.f_var)
        factor = 1 / (self.N * T * T)
        return np.array([mu, sigma]) * factor

    def calculate_susceptibility(self, m2s, m_abss, T):
        mu, sigma = bootstrapping(m_abss, m2s, self.n_boot, self.f_var)
        factor = self.N / T
        return np.array([mu, sigma]) * factor
    
    def calculate_binder(self, m2, m4):
        mu, sigma = bootstrapping(m2, m4, self.n_boot, self.f_binder)
        mu = 3/2 * (mu - 1/3)
        sigma *= 3/2
        return mu, sigma

    def calculate_correlation_length_L(self, Sq0, Sq1):
        mu, sigma = bootstrapping(Sq0, Sq1, self.n_boot, self.f_xi_L)
        factor = 1 / (2 * np.pi)
        return np.array([mu, sigma]) * factor
    
# Simulation parameters
Ls = [4, 8, 16]  # Lattice size
n_bins = 32
Tc = 2 / np.log(1 + np.sqrt(2))  # Critical temperature
n_Ts = 16
Tmin = 0.75 * Tc
Tmax = 1.25 * Tc
steps_per_temp = 2**12  # Number of Monte Carlo steps per temperature

# Results
energies = []
specific_heats = []
susceptibilities = []
Binder_cumulants = []
correlation_length_Ls = []
for L in Ls:
    ising_model = IsingSwendsenWang(L)
    E, C, chi, U2, xi_L = ising_model.simulate_temperature_range(Tmin, Tmax, n_Ts, steps_per_temp, n_bins)
    energies.append(E)
    specific_heats.append(C)
    susceptibilities.append(chi)
    Binder_cumulants.append(U2)
    correlation_length_Ls.append(xi_L)

energies = np.array(energies)
specific_heats = np.array(specific_heats)
susceptibilities = np.array(susceptibilities)
Binder_cumulants = np.array(Binder_cumulants)
correlation_length_Ls = np.array(correlation_length_Ls)

# Plotting results
plt.figure(figsize=(10, 8))

def plot_data(df):
    for i,L in enumerate(Ls):
        plt.errorbar(df[i,:,0], df[i,:,1], yerr=df[i,:,2], fmt='o-', label=r'$L=$'+str(L))
    plt.legend()
    
plt.subplot(2, 2, 1)
plt.xlabel('Temperature')
plt.ylabel('Specific heat')
plot_data(specific_heats)

plt.subplot(2, 2, 2)
plt.xlabel('Temperature')
plt.ylabel('Susceptibility')
plot_data(susceptibilities)

plt.subplot(2, 2, 3)
plt.xlabel('Temperature')
plt.ylabel('Binder cumulant')
plot_data(Binder_cumulants)

plt.subplot(2, 2, 4)
plt.xlabel('Temperature')
plt.ylabel('Correlation length / L')
plot_data(correlation_length_Ls)

plt.tight_layout()
plt.show()
