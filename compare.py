import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy import stats
import matplotlib.pyplot as plt

class RandomBitReader:
    """Quantum Random Number Generator bit reader"""
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

class QRNGNormalDist:
    """Wrapper to generate normal distribution using QRNG"""
    def __init__(self, qrng: RandomBitReader):
        self.qrng = qrng
        
    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Optional[tuple] = None) -> np.ndarray:
        """
        Generate normal distribution using Box-Muller transform
        """
        if size is None:
            size = (1,)
        elif isinstance(size, int):
            size = (size,)
            
        n_samples = np.prod(size)
        result = np.zeros(n_samples)
        
        for i in range(0, n_samples, 2):
            # Get two uniform random numbers
            u1 = self.qrng.get_random_float()
            u2 = self.qrng.get_random_float()
            
            # Box-Muller transform
            r = np.sqrt(-2.0 * np.log(u1))
            theta = 2.0 * np.pi * u2
            
            # Generate two normal random numbers
            result[i] = r * np.cos(theta)
            if i + 1 < n_samples:
                result[i + 1] = r * np.sin(theta)
        
        return loc + scale * result.reshape(size)

@dataclass
class SDEResult:
    """Container for SDE simulation results"""
    cost: float
    expectation: float
    variance: float
    third_moment: float
    fourth_moment: float
    mc_expectation: float
    mc_variance: float

def sde_qrng(l: int, N_samples: int, qrng: QRNGNormalDist, p: int = 1, 
             T: float = 0.1, sigma: float = 1.0, x0: float = 0.5) -> SDEResult:
    """
    Solve the SDE using QRNG-based normal distribution
    """
    nf = 2 ** (l-1)
    dt = T / nf
    
    X_fine = np.full(N_samples, x0)
    X_coarse = np.full(N_samples, x0)
    
    if l == 1:
        dW = np.sqrt(dt) * qrng.normal(size=N_samples)
        X_fine = X_fine - X_fine**p * dt + np.sqrt(2.0) * sigma * dW
    else:
        nc = nf // 2
        dt_coarse = T / nc
        
        for n in range(nc):
            dW0 = qrng.normal(size=N_samples)
            dW1 = qrng.normal(size=N_samples)
            
            X_fine = X_fine - X_fine**p * dt + np.sqrt(2*dt) * sigma * dW0
            X_fine = X_fine - X_fine**p * dt + np.sqrt(2*dt) * sigma * dW1
            
            dW_coarse = (dW0 + dW1) / np.sqrt(2)
            X_coarse = X_coarse - X_coarse**p * dt_coarse + np.sqrt(2*dt_coarse) * sigma * dW_coarse
    
    dP = X_fine - X_coarse if l > 1 else X_fine
    
    return SDEResult(
        cost=nf * N_samples,
        expectation=np.sum(dP),
        variance=np.sum(dP**2),
        third_moment=np.sum(dP**3),
        fourth_moment=np.sum(dP**4),
        mc_expectation=np.sum(X_fine),
        mc_variance=np.sum(X_fine**2)
    )

def mlmc_qrng(Lmin: int, Lmax: int, N0: int, eps: float, qrng: QRNGNormalDist, 
              p: int = 1) -> Tuple:
    """
    Multi-level Monte Carlo estimation using QRNG
    """
    if Lmin < 3 or Lmax < Lmin or N0 <= 0 or eps <= 0:
        raise ValueError("Invalid parameters")
    
    n_samples = np.zeros(Lmax, dtype=np.int32)
    sample_sum = np.zeros(Lmax, dtype=np.float32)
    sample_sum_squares = np.zeros(Lmax, dtype=np.float32)
    sample_mean = np.zeros(Lmax, dtype=np.float32)
    sample_var = np.zeros(Lmax, dtype=np.float32)
    
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
                result = sde_qrng(l+1, dNl[l], qrng, p)
                n_samples[l] += dNl[l]
                sample_sum[l] += result.expectation
                sample_sum_squares[l] += result.variance
                NlCl[l] += result.cost
        
        for l in range(L):
            if n_samples[l] > 0:
                sample_mean[l] = abs(sample_sum[l] / n_samples[l])
                sample_var[l] = max(
                    sample_sum_squares[l] / n_samples[l] - sample_mean[l]**2, 
                    0.0
                )
                Cl[l] = NlCl[l] / n_samples[l]
            
            if l > 1:
                sample_mean[l] = max(
                    sample_mean[l], 
                    0.5 * sample_mean[l-1] / (2**alpha)
                )
        
        s = np.sum(np.sqrt(sample_var[:L] * Cl[:L]))
        dNl[:L] = np.ceil(
            np.maximum(
                0.0, 
                np.sqrt(sample_var[:L] / Cl[:L]) * s / ((1.0 - theta) * eps**2) - n_samples[:L]
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
        
        sr = np.sum(np.maximum(0, dNl[:L] - (0.01 * n_samples[:L])))
        
        if abs(sr) < 1e-5:
            rem = sample_mean[L-1] / (2**alpha - 1.0)
            if rem <= np.sqrt(theta) * eps:
                break
            
            if L == Lmax:
                print("*** failed to achieve weak convergence ***")
                break
            
            L += 1
            sample_var[L-1] = sample_var[L-2] / (2**beta)
            Cl[L-1] = Cl[L-2] * (2**gamma)
            
            s = np.sum(np.sqrt(sample_var[:L] * Cl[:L]))
            dNl[:L] = np.ceil(
                np.maximum(
                    0.0, 
                    np.sqrt(sample_var[:L] / Cl[:L]) * s / ((1.0 - theta) * eps**2) - n_samples[:L]
                )
            ).astype(np.int32)
    
    P = np.sum(sample_sum[:L] / n_samples[:L])
    Nl[:L] = n_samples[:L]
    Cl[:L] = NlCl[:L] / Nl[:L]
    
    return P, Nl[:L], Cl[:L], sample_var[:L], alpha, beta, gamma

def sde(l: int, N_samples: int, p: int = 1, T: float = 0.1, sigma: float = 1.0, x0: float = 0.5) -> SDEResult:
    """
    Solve the SDE dX = -X^p*dt + sqrt(2)*sigma*dW using Euler-Maruyama method
    """
    nf = 2 ** (l-1)  # number of steps at fine level
    dt = T / nf
    
    # Initialize arrays
    X_fine = np.full(N_samples, x0)
    X_coarse = np.full(N_samples, x0)
    
    if l == 1:
        # Single level simulation
        dW = np.sqrt(dt) * np.random.normal(0, 1, N_samples)
        X_fine = X_fine - X_fine**p * dt + np.sqrt(2.0) * sigma * dW
    else:
        # Multilevel simulation
        nc = nf // 2
        dt_coarse = T / nc
        
        for n in range(nc):
            # Two fine steps
            dW0 = np.random.normal(0, 1, N_samples)
            dW1 = np.random.normal(0, 1, N_samples)
            
            X_fine = X_fine - X_fine**p * dt + np.sqrt(2*dt) * sigma * dW0
            X_fine = X_fine - X_fine**p * dt + np.sqrt(2*dt) * sigma * dW1
            
            # One coarse step with correlated Wiener increment
            dW_coarse = (dW0 + dW1) / np.sqrt(2)
            X_coarse = X_coarse - X_coarse**p * dt_coarse + np.sqrt(2*dt_coarse) * sigma * dW_coarse
    
    # Calculate differences and moments
    dP = X_fine - X_coarse if l > 1 else X_fine
    
    return SDEResult(
        cost=nf * N_samples,
        expectation=np.sum(dP),
        variance=np.sum(dP**2),
        third_moment=np.sum(dP**3),
        fourth_moment=np.sum(dP**4),
        mc_expectation=np.sum(X_fine),
        mc_variance=np.sum(X_fine**2)
    )


def mlmc(Lmin: int, Lmax: int, N0: int, eps: float, p: int = 1) -> Tuple:
    """
    Multi-level Monte Carlo estimation
    """
    if Lmin < 3 or Lmax < Lmin or N0 <= 0 or eps <= 0:
        raise ValueError("Invalid parameters")
    
    # Initialize arrays
    n_samples = np.zeros(Lmax, dtype=np.int32)
    sample_sum = np.zeros(Lmax, dtype=np.float32)
    sample_sum_squares = np.zeros(Lmax, dtype=np.float32)
    sample_mean = np.zeros(Lmax, dtype=np.float32)
    sample_var = np.zeros(Lmax, dtype=np.float32)
    
    L = Lmin
    theta = 0.25
    
    # Convergence parameters
    alpha = 0.0  # Expectation convergence rate
    beta = 0.0   # Variance convergence rate
    gamma = 0.0  # Cost growth rate
    
    Nl = np.zeros(Lmax, dtype=np.int32)
    Cl = np.zeros(Lmax)
    NlCl = np.zeros(Lmax)
    dNl = np.zeros(Lmax, dtype=np.int32)
    dNl[:Lmin] = N0
    
    while True:
        # Take samples at each level
        for l in range(L):
            if dNl[l] > 0:
                result = sde(l+1, dNl[l], p)
                n_samples[l] += dNl[l]
                sample_sum[l] += result.expectation
                sample_sum_squares[l] += result.variance
                NlCl[l] += result.cost
        
        # Update statistics
        for l in range(L):
            if n_samples[l] > 0:
                sample_mean[l] = abs(sample_sum[l] / n_samples[l])
                sample_var[l] = max(
                    sample_sum_squares[l] / n_samples[l] - sample_mean[l]**2, 
                    0.0
                )
                Cl[l] = NlCl[l] / n_samples[l]
            
            if l > 1:
                sample_mean[l] = max(
                    sample_mean[l], 
                    0.5 * sample_mean[l-1] / (2**alpha)
                )
        
        # Update optimal number of samples
        s = np.sum(np.sqrt(sample_var[:L] * Cl[:L]))
        dNl[:L] = np.ceil(
            np.maximum(
                0.0, 
                np.sqrt(sample_var[:L] / Cl[:L]) * s / ((1.0 - theta) * eps**2) - n_samples[:L]
            )
        ).astype(np.int32)
        
        # Estimate convergence rates
        if L > 2:
            xl = np.arange(2, L+1, dtype=float)
            yl = -np.log2(sample_mean[1:L])
            alpha = max(stats.linregress(xl, yl).slope, 0.5)
            
            yl = -np.log2(sample_var[1:L])
            beta = max(stats.linregress(xl, yl).slope, 0.5)
            
            yl = np.log2(Cl[1:L])
            gamma = max(stats.linregress(xl, yl).slope, 0.5)
        
        # Check convergence
        sr = np.sum(np.maximum(0, dNl[:L] - (0.01 * n_samples[:L])))
        
        if abs(sr) < 1e-5:
            rem = sample_mean[L-1] / (2**alpha - 1.0)
            if rem <= np.sqrt(theta) * eps:
                break
                
            if L == Lmax:
                print("*** failed to achieve weak convergence ***")
                break
                
            # Add new level
            L += 1
            sample_var[L-1] = sample_var[L-2] / (2**beta)
            Cl[L-1] = Cl[L-2] * (2**gamma)
            
            s = np.sum(np.sqrt(sample_var[:L] * Cl[:L]))
            dNl[:L] = np.ceil(
                np.maximum(
                    0.0, 
                    np.sqrt(sample_var[:L] / Cl[:L]) * s / ((1.0 - theta) * eps**2) - n_samples[:L]
                )
            ).astype(np.int32)
    
    # Calculate final estimate
    P = np.sum(sample_sum[:L] / n_samples[:L])
    Nl[:L] = n_samples[:L]
    Cl[:L] = NlCl[:L] / Nl[:L]
    
    return P, Nl[:L], Cl[:L], sample_var[:L], alpha, beta, gamma


def compare_methods():
    """Compare PRNG and QRNG MLMC methods"""
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4]
    
    # Initialize results arrays
    levels_prng = np.zeros(len(eps_values))
    costs_prng = np.zeros(len(eps_values))
    levels_qrng = np.zeros(len(eps_values))
    costs_qrng = np.zeros(len(eps_values))
    costs_sd = np.zeros(len(eps_values))
    
    # Initialize QRNG
    random_folder = Path('random')
    qrng = RandomBitReader(str(random_folder))
    qrng_normal = QRNGNormalDist(qrng)
    
    for i, eps in enumerate(eps_values):
        print(f"Running for ε = {eps}")
        
        # Run PRNG MLMC
        P_prng, Nl_prng, Cl_prng, Vl_prng, alpha_prng, beta_prng, gamma_prng = \
            mlmc(3, 10, 100, eps, p=1)
        levels_prng[i] = len(Nl_prng)
        costs_prng[i] = np.sum(Nl_prng * Cl_prng)
        
        # Run QRNG MLMC
        P_qrng, Nl_qrng, Cl_qrng, Vl_qrng, alpha_qrng, beta_qrng, gamma_qrng = \
            mlmc_qrng(3, 10, 100, eps, qrng_normal, p=1)
        levels_qrng[i] = len(Nl_qrng)
        costs_qrng[i] = np.sum(Nl_qrng * Cl_qrng)
        
        # Standard MC cost estimation
        Ntest = 10000
        result = sde(len(Nl_prng), Ntest, p=1)
        varL = max(
            result.mc_variance / Ntest - (result.mc_expectation / Ntest)**2,
            1e-10
        )
        costs_sd[i] = result.cost * varL / eps**2
    
    # Create comparison plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot levels comparison
    ax1.loglog(eps_values, levels_prng, 'bo-', label='PRNG')
    ax1.loglog(eps_values, levels_qrng, 'ro-', label='QRNG')
    ax1.set_xlabel('ε')
    ax1.set_ylabel('Levels')
    ax1.legend()
    ax1.grid(True)
    
    # Plot costs comparison
    ax2.loglog(eps_values, costs_prng, 'bo-', label='MLMC-PRNG')
    ax2.loglog(eps_values, costs_qrng, 'ro-', label='MLMC-QRNG')
    ax2.loglog(eps_values, costs_sd, 'ks--', label='Standard MC')
    ax2.set_xlabel('ε')
    ax2.set_ylabel('Cost')
    ax2.legend()
    ax2.grid(True)
    
    # Plot speedup comparison
    speedup_prng = costs_sd / costs_prng
    speedup_qrng = costs_sd / costs_qrng
    ax3.loglog(eps_values, speedup_prng, 'bo-', label='PRNG Speedup')
    ax3.loglog(eps_values, speedup_qrng, 'ro-', label='QRNG Speedup')
    ax3.set_xlabel('ε')
    ax3.set_ylabel('Speed-up')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return speedup_prng, speedup_qrng

if __name__ == "__main__":
    compare_methods()