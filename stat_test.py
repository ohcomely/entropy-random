import numpy as np
from numpy.random import MT19937, Generator
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

def generate_binary_matrix(rng, size):
    """Generate a square binary matrix using the given RNG."""
    # Generate random integers and convert to binary
    matrix = rng.integers(0, 2, size=(size, size), dtype=np.int8)
    return matrix

def compute_ranks(rng, matrix_size, num_matrices):
    """Compute ranks for multiple binary matrices."""
    ranks = []
    for _ in range(num_matrices):
        matrix = generate_binary_matrix(rng, matrix_size)
        # Compute rank over F2 (GF(2))
        rank = matrix_rank(matrix % 2)
        ranks.append(rank)
    return ranks

def plot_rank_distribution(mt_ranks, random_ranks, matrix_size):
    """Plot histogram of matrix ranks for comparison."""
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(mt_ranks, bins=range(matrix_size-5, matrix_size+2), alpha=0.5, 
            label='Mersenne Twister', color='red')
    plt.hist(random_ranks, bins=range(matrix_size-5, matrix_size+2), alpha=0.5,
            label='System Random', color='blue')
    
    plt.title(f'Binary Matrix Rank Distribution (Size: {matrix_size}x{matrix_size})')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Print statistics
    print(f"Mersenne Twister - Mean rank: {np.mean(mt_ranks):.2f}")
    print(f"System Random - Mean rank: {np.mean(random_ranks):.2f}")

def main():
    # Parameters
    MATRIX_SIZE = 300
    NUM_MATRICES = 1000
    
    # Initialize PRNGs
    mt_rng = Generator(MT19937())
    random_rng = Generator(np.random.PCG64())  # Using PCG64 as a higher-quality PRNG
    
    # Generate and compute ranks
    mt_ranks = compute_ranks(mt_rng, MATRIX_SIZE, NUM_MATRICES)
    random_ranks = compute_ranks(random_rng, MATRIX_SIZE, NUM_MATRICES)
    
    # Plot results
    plot_rank_distribution(mt_ranks, random_ranks, MATRIX_SIZE)
    plt.show()

if __name__ == "__main__":
    main()