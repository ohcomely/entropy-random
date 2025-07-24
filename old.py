import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def bits_to_uniform(bits, num_points):
    """Convert bit string to uniform random numbers in [0,1]"""
    # Use 32 bits per number for good precision
    bits_per_num = 32
    numbers = []
    for i in range(0, min(len(bits) - bits_per_num, num_points * bits_per_num), bits_per_num):
        # Convert 32 bits to integer then normalize to [0,1]
        val = int(bits[i:i+bits_per_num], 2)
        numbers.append(val / (2**bits_per_num))
    return numbers[:num_points]

def load_qrng_data(file_nums, points_needed):
    """Load and process QRNG data from multiple files"""
    all_bits = ""
    for i in file_nums:
        with open(f'random/{i}.txt', 'r') as f:
            all_bits += f.read().strip()
            if len(all_bits) >= points_needed * 32:  # 32 bits per number
                break
    return all_bits

class PiEstimator:
    def __init__(self, M, N, use_qrng=False, file_nums=None):
        self.M = M  # points per sample
        self.N = N  # number of samples
        self.use_qrng = use_qrng
        self.file_nums = file_nums
        
        if use_qrng:
            # Load QRNG data
            bits_needed = M * N * 32 * 2  # *2 for x and y coordinates
            self.qrng_bits = load_qrng_data(file_nums, M * N * 2)
        
    def method_a_sample(self, sample_idx):
        """Monte Carlo estimation of π/4 using point in circle method"""
        if self.use_qrng:
            start_idx = sample_idx * self.M * 2
            x = np.array(bits_to_uniform(self.qrng_bits[start_idx*32:(start_idx+self.M)*32], self.M))
            y = np.array(bits_to_uniform(self.qrng_bits[(start_idx+self.M)*32:(start_idx+2*self.M)*32], self.M))
        else:
            x = np.random.random(self.M)
            y = np.random.random(self.M)
            
        inside = sum(x*x + y*y <= 1)
        return 4 * inside / self.M


    def method_b_sample(self, sample_idx):
        """Buffon's needle experiment"""
        if self.use_qrng:
            start_idx = sample_idx * self.M * 2
            rho = np.array(bits_to_uniform(self.qrng_bits[start_idx*32:(start_idx+self.M)*32], self.M)) * 0.5
            theta = np.array(bits_to_uniform(self.qrng_bits[(start_idx+self.M)*32:(start_idx+2*self.M)*32], self.M)) * (np.pi / 2)
        else:
            rho = np.random.random(self.M) * 0.5
            theta = np.random.random(self.M) * np.pi/2
            
        crossings = sum(r < 0.5 * np.sin(t) for r, t in zip(rho, theta))
        return 2 * self.M / crossings if crossings > 0 else float('inf')


    def run_estimation(self, method='A'):
        estimates = []
        method_func = self.method_a_sample if method == 'A' else self.method_b_sample
        
        for i in tqdm(range(self.N), desc=f'Method {method} {"QRNG" if self.use_qrng else "PRNG"}'):
            pi_est = method_func(i)
            if pi_est != float('inf'):
                estimates.append(pi_est)
                
        return np.array(estimates)

def analyze_results(prng_results, qrng_results, M, method, show_plots=True):
    """Analyze and plot results"""
    def calc_error(estimates):
        return np.abs(estimates - np.pi)
    
    prng_errors = calc_error(prng_results)
    qrng_errors = calc_error(qrng_results)
    
    # Calculate statistics
    stats = {
        'PRNG': {
            'mean': np.mean(prng_results),
            'std': np.std(prng_results),
            'mean_error': np.mean(prng_errors),
            'median_error': np.median(prng_errors)
        },
        'QRNG': {
            'mean': np.mean(qrng_results),
            'std': np.std(qrng_results),
            'mean_error': np.mean(qrng_errors),
            'median_error': np.median(qrng_errors)
        }
    }
    
    if show_plots:
        # Create line plot for absolute error vs. number of points
        plt.figure(figsize=(12, 6))
        
        x_points = np.arange(1, len(prng_errors) + 1)
        
        plt.plot(x_points, prng_errors, label='PRNG Errors', marker='o', linestyle='-', alpha=0.7)
        plt.plot(x_points, qrng_errors, label='QRNG Errors', marker='o', linestyle='-', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
        
        plt.title(f'Absolute Error vs. Number of Points (Method {method})\nM={M}')
        plt.xlabel('Number of Points (Cumulative)')
        plt.ylabel('Absolute Error |Estimate - π|')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    return stats

# Main execution
M = 100000  # points per sample
N = 100  # number of samples
file_nums = range(1, 154)  # all available files

# Run Method A
estimator_prng_a = PiEstimator(M, N, use_qrng=False)
estimator_qrng_a = PiEstimator(M, N, use_qrng=True, file_nums=file_nums)

results_a_prng = estimator_prng_a.run_estimation('A')
results_a_qrng = estimator_qrng_a.run_estimation('A')

# Run Method B
estimator_prng_b = PiEstimator(M, N, use_qrng=False)
estimator_qrng_b = PiEstimator(M, N, use_qrng=True, file_nums=file_nums)

results_b_prng = estimator_prng_b.run_estimation('B')
results_b_qrng = estimator_qrng_b.run_estimation('B')

# Analyze and plot results
print("\nMethod A Results:")
stats_a = analyze_results(results_a_prng, results_a_qrng, M, 'A')
print("\nMethod B Results:")
stats_b = analyze_results(results_b_prng, results_b_qrng, M, 'B')

# Print detailed statistics
for method, stats in [("A", stats_a), ("B", stats_b)]:
    print(f"\nDetailed Statistics for Method {method}:")
    print("\nPRNG:")
    print(f"Mean π estimate: {stats['PRNG']['mean']:.6f}")
    print(f"Standard deviation: {stats['PRNG']['std']:.6f}")
    print(f"Mean absolute error: {stats['PRNG']['mean_error']:.6f}")
    print(f"Median absolute error: {stats['PRNG']['median_error']:.6f}")
    
    print("\nQRNG:")
    print(f"Mean π estimate: {stats['QRNG']['mean']:.6f}")
    print(f"Standard deviation: {stats['QRNG']['std']:.6f}")
    print(f"Mean absolute error: {stats['QRNG']['mean_error']:.6f}")
    print(f"Median absolute error: {stats['QRNG']['median_error']:.6f}")