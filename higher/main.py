import numpy as np
from pathlib import Path
import time
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm

class RandomBitReader:
    def __init__(self, folder_path):
        self.files = sorted(Path(folder_path).glob('*.txt'))
        self.current_file_idx = 0
        self.current_bits = ""
        self.bit_position = 0
        self.total_bits_read = 0
        self._load_next_file()
        
    def _load_next_file(self):
        if self.current_file_idx < len(self.files):
            with open(self.files[self.current_file_idx], 'r') as f:
                self.current_bits = f.read().strip()
            self.bit_position = 0
            self.current_file_idx += 1
            return True
        return False
        
    def get_bits(self, n):
        result = ""
        while len(result) < n:
            if self.bit_position >= len(self.current_bits):
                if not self._load_next_file():
                    raise RuntimeError(f"Ran out of random bits. Total bits read: {self.total_bits_read}")
            available = min(n - len(result), len(self.current_bits) - self.bit_position)
            result += self.current_bits[self.bit_position:self.bit_position + available]
            self.bit_position += available
            self.total_bits_read += available
        return result
        
    def get_random_float(self):
        bits = self.get_bits(24)
        value = int(bits, 2)
        return value / (2**24)
    
    def get_random_array(self, size):
        return np.array([self.get_random_float() for _ in range(size)])

def true_hypersphere_volume(n):
    """Calculate the true volume of an n-dimensional hypersphere with radius 1"""
    return (np.pi ** (n/2)) / gamma(n/2 + 1)

def estimate_volume_mt_improved(dimension, num_points):
    """Improved estimator using Mersenne Twister"""
    points = np.random.uniform(0, 1, (num_points, dimension))
    distances = np.sqrt(np.sum(points**2, axis=1))
    
    # Use distance-based estimator
    volume_estimate = (2**dimension) * np.mean(distances <= 1)
    
    # Calculate standard error
    hits = (distances <= 1)
    std_error = np.std(hits) / np.sqrt(num_points)
    
    return volume_estimate, std_error

def estimate_volume_qrng_improved(dimension, num_points, reader):
    """Improved estimator using QRNG"""
    distances = []
    hits = []
    
    for _ in range(num_points):
        point = reader.get_random_array(dimension)
        distance = np.sqrt(np.sum(point**2))
        distances.append(distance)
        hits.append(distance <= 1)
    
    distances = np.array(distances)
    hits = np.array(hits)
    
    # Use distance-based estimator
    volume_estimate = (2**dimension) * np.mean(hits)
    
    # Calculate standard error
    std_error = np.std(hits) / np.sqrt(num_points)
    
    return volume_estimate, std_error

# Test parameters
dimensions = [5, 7, 10, 12, 15, 20]
num_trials = 10
sample_sizes = [5000, 10000, 25000, 50000]

# Initialize results storage
results = {
    'mt': {size: {dim: {'errors': [], 'std_errors': []} for dim in dimensions} for size in sample_sizes},
    'qrng': {size: {dim: {'errors': [], 'std_errors': []} for dim in dimensions} for size in sample_sizes},
    'true_volumes': {dim: true_hypersphere_volume(dim) for dim in dimensions}
}

# Run experiments
# reader = RandomBitReader("../random")

for size in tqdm(sample_sizes, desc="Sample sizes"):
    for dim in tqdm(dimensions, desc=f"Dimensions (n={size})", leave=False):
        true_vol = results['true_volumes'][dim]
        
        for trial in range(num_trials):
            # Mersenne Twister
            mt_vol, mt_std = estimate_volume_mt_improved(dim, size)
            rel_error = abs(mt_vol - true_vol) / true_vol
            results['mt'][size][dim]['errors'].append(rel_error)
            results['mt'][size][dim]['std_errors'].append(mt_std)
            
            
            # QRNG
            reader = RandomBitReader("../random")
            qrng_vol, qrng_std = estimate_volume_qrng_improved(dim, size, reader)
            rel_error = abs(qrng_vol - true_vol) / true_vol
            results['qrng'][size][dim]['errors'].append(rel_error)
            results['qrng'][size][dim]['std_errors'].append(qrng_std)

# Create a figure with two subplots side by side
plt.figure(figsize=(20, 8))

# First subplot - Original comparison plot
plt.subplot(1, 2, 1)
markers = ['o', 's', '^', 'D']
colors = ['blue', 'red']
linestyles = ['-', '--']

for idx, size in enumerate(sample_sizes):
    # Calculate mean relative errors and standard errors
    mt_errors = [np.mean(results['mt'][size][dim]['errors']) for dim in dimensions]
    qrng_errors = [np.mean(results['qrng'][size][dim]['errors']) for dim in dimensions]
    
    mt_stderr = [np.mean(results['mt'][size][dim]['std_errors']) for dim in dimensions]
    qrng_stderr = [np.mean(results['qrng'][size][dim]['std_errors']) for dim in dimensions]
    
    # Plot MT results with error bars
    plt.errorbar(dimensions, mt_errors, yerr=mt_stderr,
                marker=markers[idx], color=colors[0], linestyle=linestyles[0],
                label=f'MT (n={size})', capsize=5)
    
    # Plot QRNG results with error bars
    plt.errorbar(dimensions, qrng_errors, yerr=qrng_stderr,
                marker=markers[idx], color=colors[1], linestyle=linestyles[1],
                label=f'QRNG (n={size})', capsize=5)

plt.yscale('log')
plt.xlabel('Dimension')
plt.ylabel('Mean Relative Error (log scale)')
plt.title('MT vs QRNG Error by Dimension')
plt.legend()
plt.grid(True)

# Second subplot - Convergence plot
plt.subplot(1, 2, 2)
selected_dims = [5, 10, 15, 20]  # Select a subset of dimensions for clarity
markers = ['o', 's', '^', 'D', '*']
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_dims)))

# Theoretical Monte Carlo convergence rate (1/√N)
theoretical_convergence = [1/np.sqrt(n) for n in sample_sizes]
plt.plot(sample_sizes, theoretical_convergence, 'k--', label='Theoretical (1/√N)', alpha=0.5)

for idx, dim in enumerate(selected_dims):
    # MT convergence
    mt_errors = [np.mean(results['mt'][size][dim]['errors']) for size in sample_sizes]
    plt.plot(sample_sizes, mt_errors, color=colors[idx], linestyle='-', marker='o',
             label=f'MT (d={dim})')
    
    # QRNG convergence
    qrng_errors = [np.mean(results['qrng'][size][dim]['errors']) for size in sample_sizes]
    plt.plot(sample_sizes, qrng_errors, color=colors[idx], linestyle='--', marker='s',
             label=f'QRNG (d={dim})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Sample Size (log scale)')
plt.ylabel('Mean Relative Error (log scale)')
plt.title('Convergence Analysis by Sample Size')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison.png', transparent=True)
plt.show()

# Print convergence rates
print("\nConvergence Rate Analysis:")
print("\nDim | Method | Empirical Rate")
print("-" * 35)

for dim in selected_dims:
    # Calculate empirical convergence rates using linear regression on log-log scale
    log_samples = np.log(sample_sizes)
    
    # MT convergence rate
    log_mt_errors = np.log([np.mean(results['mt'][size][dim]['errors']) for size in sample_sizes])
    mt_rate = np.polyfit(log_samples, log_mt_errors, 1)[0]
    
    # QRNG convergence rate
    log_qrng_errors = np.log([np.mean(results['qrng'][size][dim]['errors']) for size in sample_sizes])
    qrng_rate = np.polyfit(log_samples, log_qrng_errors, 1)[0]
    
    print(f"{dim:3d} | MT    | {mt_rate:.7f}")
    print(f"{dim:3d} | QRNG  | {qrng_rate:.7f}")