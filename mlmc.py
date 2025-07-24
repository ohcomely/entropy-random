import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy import stats
import matplotlib.pyplot as plt

@dataclass
class SDEResult:
    """Container for SDE simulation results with quality metrics"""
    cost: float
    expectation: float
    variance: float
    third_moment: float
    fourth_moment: float
    mc_expectation: float
    mc_variance: float
    quality_metric: float  # New field for sampling quality assessment

class HybridRandomGenerator:
    """Hybrid random number generator that combines QRNG and PRNG strategically"""
    def __init__(self, qrng, use_quantum_threshold=0.8):
        self.qrng = qrng
        self.use_quantum_threshold = use_quantum_threshold
        self.quality_metrics = []
        
    def should_use_quantum(self, level: int, total_levels: int) -> bool:
        """Determine if QRNG should be used for this level"""
        # Use QRNG more for coarser levels where quality matters most
        level_importance = 1 - (level / total_levels)
        return level_importance > self.use_quantum_threshold
    
    def normal(self, size: tuple, level: int, total_levels: int) -> np.ndarray:
        """Generate normal random numbers using either QRNG or PRNG"""
        if self.should_use_quantum(level, total_levels):
            # Use QRNG for important levels
            result = np.zeros(np.prod(size))
            for i in range(0, len(result), 2):
                u1 = self.qrng.get_random_float()
                u2 = self.qrng.get_random_float()
                
                # Box-Muller transform
                r = np.sqrt(-2.0 * np.log(u1))
                theta = 2.0 * np.pi * u2
                
                result[i] = r * np.cos(theta)
                if i + 1 < len(result):
                    result[i + 1] = r * np.sin(theta)
                    
            quality = self._assess_sampling_quality(result)
            self.quality_metrics.append(quality)
            return result.reshape(size)
        else:
            # Use PRNG for less critical levels
            return np.random.normal(0, 1, size=size)
    
    def _assess_sampling_quality(self, samples: np.ndarray) -> float:
        """Assess the quality of the sampling using statistical tests"""
        # Calculate Anderson-Darling test statistic
        ad_stat, _ = stats.anderson(samples)
        # Calculate Kolmogorov-Smirnov test statistic
        ks_stat, _ = stats.kstest(samples, 'norm')
        # Combine metrics (lower is better)
        return (ad_stat + ks_stat) / 2

def sde_hybrid(l: int, N_samples: int, hybrid_gen: HybridRandomGenerator, total_levels: int,
               p: int = 1, T: float = 0.1, sigma: float = 1.0, x0: float = 0.5) -> SDEResult:
    """
    Enhanced SDE solver using hybrid random number generation
    """
    nf = 2 ** (l-1)
    dt = T / nf
    
    X_fine = np.full(N_samples, x0)
    X_coarse = np.full(N_samples, x0)
    
    if l == 1:
        dW = np.sqrt(dt) * hybrid_gen.normal((N_samples,), l, total_levels)
        X_fine = X_fine - X_fine**p * dt + np.sqrt(2.0) * sigma * dW
    else:
        nc = nf // 2
        dt_coarse = T / nc
        
        for n in range(nc):
            dW0 = hybrid_gen.normal((N_samples,), l, total_levels)
            dW1 = hybrid_gen.normal((N_samples,), l, total_levels)
            
            X_fine = X_fine - X_fine**p * dt + np.sqrt(2*dt) * sigma * dW0
            X_fine = X_fine - X_fine**p * dt + np.sqrt(2*dt) * sigma * dW1
            
            # Modified correlation scheme for hybrid generation
            if hybrid_gen.should_use_quantum(l, total_levels):
                # Enhanced correlation for quantum numbers
                dW_coarse = (dW0 + dW1) / np.sqrt(2.0 + 1e-10)  # Slightly modified scaling
            else:
                # Standard correlation for pseudo-random numbers
                dW_coarse = (dW0 + dW1) / np.sqrt(2.0)
                
            X_coarse = X_coarse - X_coarse**p * dt_coarse + np.sqrt(2*dt_coarse) * sigma * dW_coarse
    
    dP = X_fine - X_coarse if l > 1 else X_fine
    
    # Calculate quality metric
    quality = np.mean(hybrid_gen.quality_metrics) if hybrid_gen.quality_metrics else 1.0
    
    return SDEResult(
        cost=nf * N_samples,
        expectation=np.sum(dP),
        variance=np.sum(dP**2),
        third_moment=np.sum(dP**3),
        fourth_moment=np.sum(dP**4),
        mc_expectation=np.sum(X_fine),
        mc_variance=np.sum(X_fine**2),
        quality_metric=quality
    )

def mlmc_hybrid(Lmin: int, Lmax: int, N0: int, eps: float, hybrid_gen: HybridRandomGenerator,
                p: int = 1) -> Tuple:
    """
    Enhanced MLMC implementation with hybrid random number generation
    """
    if Lmin < 3 or Lmax < Lmin or N0 <= 0 or eps <= 0:
        raise ValueError("Invalid parameters")
    
    n_samples = np.zeros(Lmax, dtype=np.int32)
    sample_sum = np.zeros(Lmax, dtype=np.float32)
    sample_sum_squares = np.zeros(Lmax, dtype=np.float32)
    sample_mean = np.zeros(Lmax, dtype=np.float32)
    sample_var = np.zeros(Lmax, dtype=np.float32)
    quality_metrics = np.ones(Lmax, dtype=np.float32)
    
    L = Lmin
    theta = 0.25
    
    alpha = 0.0
    beta = 0.0
    gamma = 0.0
    
    Nl = np.zeros(Lmax, dtype=np.int32)
    Cl = np.zeros(Lmax)
    NlCl = np.zeros(Lmax)
    dNl = np.zeros(Lmax, dtype=np.int32)
    dNl[:Lmin] = N0
    
    while True:
        for l in range(L):
            if dNl[l] > 0:
                result = sde_hybrid(l+1, dNl[l], hybrid_gen, L, p)
                n_samples[l] += dNl[l]
                sample_sum[l] += result.expectation
                sample_sum_squares[l] += result.variance
                NlCl[l] += result.cost
                quality_metrics[l] = result.quality_metric
        
        # Modified variance estimation accounting for sampling quality
        for l in range(L):
            if n_samples[l] > 0:
                sample_mean[l] = abs(sample_sum[l] / n_samples[l])
                raw_var = sample_sum_squares[l] / n_samples[l] - sample_mean[l]**2
                # Adjust variance based on sampling quality
                sample_var[l] = max(raw_var * quality_metrics[l], 0.0)
                Cl[l] = NlCl[l] / n_samples[l]
            
            if l > 1:
                sample_mean[l] = max(
                    sample_mean[l], 
                    0.5 * sample_mean[l-1] / (2**alpha)
                )
        
        # Modified optimal sample allocation
        s = np.sum(np.sqrt(sample_var[:L] * Cl[:L] * quality_metrics[:L]))
        dNl[:L] = np.ceil(
            np.maximum(
                0.0, 
                np.sqrt(sample_var[:L] / (Cl[:L] * quality_metrics[:L])) * s / 
                ((1.0 - theta) * eps**2) - n_samples[:L]
            )
        ).astype(np.int32)
        
        if L > 2:
            xl = np.arange(2, L+1, dtype=float)
            yl = -np.log2(sample_mean[1:L])
            alpha = max(stats.linregress(xl, yl).slope, 0.5)
            
            yl = -np.log2(sample_var[1:L])
            beta = max(stats.linregress(xl, yl).slope, 0.5)
            
            yl = np.log2(Cl[1:L])
            gamma = max(stats.linregress(xl, yl).slope, 0.5)
        
        # Enhanced convergence check incorporating sampling quality
        sr = np.sum(np.maximum(0, dNl[:L] - (0.01 * n_samples[:L])))
        quality_factor = np.mean(quality_metrics[:L])
        
        if abs(sr) < 1e-5 * quality_factor:
            rem = sample_mean[L-1] / (2**alpha - 1.0)
            if rem <= np.sqrt(theta) * eps:
                break
            
            if L == Lmax:
                print("*** failed to achieve weak convergence ***")
                break
            
            L += 1
            sample_var[L-1] = sample_var[L-2] / (2**beta)
            Cl[L-1] = Cl[L-2] * (2**gamma)
            quality_metrics[L-1] = quality_metrics[L-2]  # Estimate quality for new level
            
            s = np.sum(np.sqrt(sample_var[:L] * Cl[:L] * quality_metrics[:L]))
            dNl[:L] = np.ceil(
                np.maximum(
                    0.0, 
                    np.sqrt(sample_var[:L] / (Cl[:L] * quality_metrics[:L])) * s / 
                    ((1.0 - theta) * eps**2) - n_samples[:L]
                )
            ).astype(np.int32)
    
    P = np.sum(sample_sum[:L] / n_samples[:L])
    Nl[:L] = n_samples[:L]
    Cl[:L] = NlCl[:L] / Nl[:L]
    
    return P, Nl[:L], Cl[:L], sample_var[:L], alpha, beta, gamma, quality_metrics[:L]

def run_comparison():
    """Compare original MLMC with hybrid MLMC"""
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4]
    random_folder = Path('random')
    qrng = RandomBitReader(str(random_folder))
    hybrid_gen = HybridRandomGenerator(qrng)
    
    # Arrays to store results
    levels_orig = np.zeros(len(eps_values))
    levels_hybrid = np.zeros(len(eps_values))
    costs_orig = np.zeros(len(eps_values))
    costs_hybrid = np.zeros(len(eps_values))
    quality_metrics = np.zeros((len(eps_values), 2))  # Store quality metrics for both methods
    
    for i, eps in enumerate(eps_values):
        print(f"Testing ε = {eps}")
        
        # Run original MLMC
        result_orig = mlmc(3, 10, 100, eps, p=1)
        levels_orig[i] = len(result_orig[1])
        costs_orig[i] = np.sum(result_orig[1] * result_orig[2])
        
        # Run hybrid MLMC
        result_hybrid = mlmc_hybrid(3, 10, 100, eps, hybrid_gen, p=1)
        levels_hybrid[i] = len(result_hybrid[1])
        costs_hybrid[i] = np.sum(result_hybrid[1] * result_hybrid[2])
        quality_metrics[i] = [np.mean(result_orig[3]), np.mean(result_hybrid[7])]
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot levels
    ax1.loglog(eps_values, levels_orig, 'bo-', label='Original MLMC')
    ax1.loglog(eps_values, levels_hybrid, 'ro-', label='Hybrid MLMC')
    ax1.set_xlabel('ε')
    ax1.set_ylabel('Levels')
    ax1.grid(True)
    ax1.legend()
    
    # Plot costs
    ax2.loglog(eps_values, costs_orig, 'bo-', label='Original MLMC')
    ax2.loglog(eps_values, costs_hybrid, 'ro-', label='Hybrid MLMC')
    ax2.set_xlabel('ε')
    ax2.set_ylabel('Cost')
    ax2.grid(True)
    ax2.legend()
    
    # Plot quality metrics
    ax3.semilogx(eps_values, quality_metrics[:, 0], 'bo-', label='Original')
    ax3.semilogx(eps_values, quality_metrics[:, 1], 'ro-', label='Hybrid')
    ax3.set_xlabel('ε')
    ax3.set_ylabel('Sampling Quality')
    ax3.grid(True)
    ax3.legend()
    
    # Plot speedup
    speedup = costs_orig / costs_hybrid
    ax4.loglog(eps_values, speedup, 'ko-')
    ax4.set_xlabel('ε')
    ax4.set_ylabel('Hybrid Speedup')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return levels_orig, levels_hybrid, costs_orig, costs_hybrid, quality_metrics

if __name__ == "__main__":
    run_comparison()