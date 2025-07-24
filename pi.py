import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
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
        # Use 24 bits for single precision float
        bits = self.get_bits(24)
        value = int(bits, 2)
        return value / (2**24)

def calculate_required_bits(n_samples, points_per_sample):
    """Calculate total bits needed for the experiment"""
    bits_per_coord = 24
    coords_per_point = 2
    return n_samples * points_per_sample * bits_per_coord * coords_per_point

def method_a_sample(rng, points_per_sample):
    """π/4 approximation method"""
    if isinstance(rng, np.random.Generator):
        points = rng.random((points_per_sample, 2))
    else:
        points = np.array([[rng.get_random_float() for _ in range(2)] 
                          for _ in range(points_per_sample)])
    
    inside_circle = np.sum(np.sqrt(np.sum(points**2, axis=1)) <= 1)
    return 4 * inside_circle / points_per_sample

def method_b_sample(rng, points_per_sample):
    """Buffon's needle method"""
    if isinstance(rng, np.random.Generator):
        rho = rng.random(points_per_sample) / 2
        theta = rng.random(points_per_sample) * np.pi/2
    else:
        rho = np.array([rng.get_random_float() for _ in range(points_per_sample)]) / 2
        theta = np.array([rng.get_random_float() for _ in range(points_per_sample)]) * np.pi/2
    
    crosses = np.sum(rho < np.sin(theta)/2)
    return 2 / (crosses / points_per_sample) if crosses > 0 else np.inf

def run_experiment(rng, method, n_samples, points_per_sample):
    results = []
    for _ in tqdm(range(n_samples), desc=f"Running {method.__name__}"):
        pi_approx = method(rng, points_per_sample)
        results.append(pi_approx)
    return np.array(results)

def plot_convergence(results_prng, results_qrng, method_name, points_per_sample):
    plt.figure(figsize=(12, 8))
    
    # Calculate empirical means
    x = np.arange(1, len(results_prng) + 1) * points_per_sample
    means_prng = np.cumsum(results_prng) / np.arange(1, len(results_prng) + 1)
    means_qrng = np.cumsum(results_qrng) / np.arange(1, len(results_qrng) + 1)
    
    # Calculate confidence intervals (95%)
    std_prng = np.array([np.std(results_prng[:i+1]) / np.sqrt(i+1) 
                        for i in range(len(results_prng))])
    std_qrng = np.array([np.std(results_qrng[:i+1]) / np.sqrt(i+1) 
                        for i in range(len(results_qrng))])
    
    ci_prng = 1.96 * std_prng
    ci_qrng = 1.96 * std_qrng
    
    # Plot setup
    plt.semilogx(x, means_prng, 'r-', label='PRNG', alpha=0.8)
    plt.semilogx(x, means_qrng, 'b-', label='QRNG', alpha=0.8)
    
    # Plot confidence intervals
    plt.fill_between(x, means_prng - ci_prng, means_prng + ci_prng, 
                    color='r', alpha=0.2)
    plt.fill_between(x, means_qrng - ci_qrng, means_qrng + ci_qrng, 
                    color='b', alpha=0.2)
    
    # Add reference lines
    plt.axhline(y=np.pi, color='k', linestyle='--', label='π')
    plt.axhline(y=np.mean(results_prng), color='r', linestyle=':', 
                label='PRNG mean', alpha=0.5)
    plt.axhline(y=np.mean(results_qrng), color='b', linestyle=':', 
                label='QRNG mean', alpha=0.5)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title(f'Empirical means of {method_name}, M = {points_per_sample}')
    plt.xlabel('Sample size N')
    plt.ylabel('Estimated value of π')
    plt.legend()
    
    # Format y-axis to show more decimal places
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    plt.tight_layout()
    plt.savefig(f'convergence_{method_name.replace(" ", "_")}.png', dpi=300)
    plt.close()

def plot_results(results_prng, results_qrng, method_name):
    plt.figure(figsize=(15, 10))
    
    # Distribution plot
    plt.subplot(2, 2, 1)
    sns.histplot(data=pd.DataFrame({
        'PRNG': results_prng,
        'QRNG': results_qrng
    }).melt(), x='value', hue='variable', stat='density', common_norm=False)
    plt.axvline(np.pi, color='r', linestyle='--', label='π')
    plt.title(f'{method_name} - Distribution of π Approximations')
    plt.legend()
    
    # Error convergence plot
    plt.subplot(2, 2, 2)
    errors_prng = np.abs(np.cumsum(results_prng) / np.arange(1, len(results_prng) + 1) - np.pi)
    errors_qrng = np.abs(np.cumsum(results_qrng) / np.arange(1, len(results_qrng) + 1) - np.pi)
    
    plt.loglog(np.arange(1, len(errors_prng) + 1), errors_prng, label='PRNG')
    plt.loglog(np.arange(1, len(errors_qrng) + 1), errors_qrng, label='QRNG')
    plt.loglog(np.arange(1, len(errors_prng) + 1), 
              1/np.sqrt(np.arange(1, len(errors_prng) + 1)), 
              '--', label='1/√N reference')
    plt.title('Error Convergence')
    plt.xlabel('Number of samples')
    plt.ylabel('Absolute Error')
    plt.legend()
    
    # Statistical tests
    plt.subplot(2, 2, 3)
    plt.text(0.1, 0.9, f"PRNG mean: {np.mean(results_prng):.6f}")
    plt.text(0.1, 0.8, f"QRNG mean: {np.mean(results_qrng):.6f}")
    plt.text(0.1, 0.7, f"PRNG std: {np.std(results_prng):.6f}")
    plt.text(0.1, 0.6, f"QRNG std: {np.std(results_qrng):.6f}")
    
    # KS test
    ks_stat, ks_pval = stats.ks_2samp(results_prng, results_qrng)
    plt.text(0.1, 0.4, f"KS test p-value: {ks_pval:.6f}")
    
    # t-test
    t_stat, t_pval = stats.ttest_ind(results_prng, results_qrng)
    plt.text(0.1, 0.3, f"T-test p-value: {t_pval:.6f}")
    plt.axis('off')
    
    # Q-Q plot
    plt.subplot(2, 2, 4)
    stats.probplot(results_prng, dist="norm", plot=plt)
    plt.title('PRNG Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(f'pi_approximation_{method_name.replace(" ", "_")}.png')
    plt.close()

def main():
    # Parameters for experiment
    N_SAMPLES = 1000 
    POINTS_PER_SAMPLE = 20000

    # Calculate required bits
    required_bits = calculate_required_bits(N_SAMPLES, POINTS_PER_SAMPLE)
    print(f"Required bits for experiment: {required_bits:,}")

    # # Initialize random number generators
    # mt19937 = np.random.MT19937(42)  # Create MT19937 instance
    # prng = np.random.Generator(mt19937)  # Create Generator from MT19937
    # random_folder = Path('random')
    # print(f"Found {len(list(random_folder.glob('*.txt')))} random bit files")

    # # Estimate available bits (assuming 8MB per file)
    # available_bits = len(list(random_folder.glob('*.txt'))) * 8_388_608
    # print(f"Estimated available bits: {available_bits:,}")

    # if required_bits > available_bits:
    #     print("Warning: Required bits exceed available bits. Adjusting parameters...")
    #     scale_factor = available_bits / required_bits / 2  # Using half available bits for safety
    #     N_SAMPLES = int(N_SAMPLES * scale_factor)
    #     POINTS_PER_SAMPLE = max(50, int(POINTS_PER_SAMPLE * scale_factor))
    #     print(f"Adjusted parameters: N_SAMPLES={N_SAMPLES}, POINTS_PER_SAMPLE={POINTS_PER_SAMPLE}")

    # qrng = RandomBitReader(str(random_folder))
    
    # # Run experiments
    # print("\nRunning Method A (π/4 approximation)...")
    # results_a_prng = run_experiment(prng, method_a_sample, N_SAMPLES, POINTS_PER_SAMPLE)
    # results_a_qrng = run_experiment(qrng, method_a_sample, N_SAMPLES, POINTS_PER_SAMPLE)
    # plot_results(results_a_prng, results_a_qrng, "Method A")
    # plot_convergence(results_a_prng, results_a_qrng, "Method A", POINTS_PER_SAMPLE)
    
    # print("\nRunning Method B (Buffon's needle)...")
    # qrng = RandomBitReader(str(random_folder))  # Reset QRNG
    # results_b_prng = run_experiment(prng, method_b_sample, N_SAMPLES, POINTS_PER_SAMPLE)
    # results_b_qrng = run_experiment(qrng, method_b_sample, N_SAMPLES, POINTS_PER_SAMPLE)
    # plot_results(results_b_prng, results_b_qrng, "Method B")
    # plot_convergence(results_b_prng, results_b_qrng, "Method B", POINTS_PER_SAMPLE)
    
    # # Save results to CSV
    # pd.DataFrame({
    #     'Method_A_PRNG': results_a_prng,
    #     'Method_A_QRNG': results_a_qrng,
    #     'Method_B_PRNG': results_b_prng,
    #     'Method_B_QRNG': results_b_qrng
    # }).to_csv('pi_approximation_results.csv', index=False)

if __name__ == "__main__":
    main()