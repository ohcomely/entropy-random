import numpy as np
from collections import Counter
import math
from pathlib import Path
import matplotlib.pyplot as plt
import tqdm

# Function to calculate entropy
def entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

# Function to calculate conditional entropy
def conditional_entropy(sequence, macrostate_bins):
    """
    Calculates conditional entropy for a sequence of random numbers.
    :param sequence: The input random number sequence (list of floats).
    :param macrostate_bins: Number of macrostates to partition [0, 1) into.
    :return: Conditional entropy.
    """
    bin_edges = np.linspace(0, 1, macrostate_bins + 1)
    macrostates = np.digitize(sequence, bin_edges) - 1

    # Count macrostate transitions
    transitions = Counter((macrostates[i], macrostates[i + 1]) for i in range(len(macrostates) - 1))
    total_transitions = sum(transitions.values())

    # Calculate joint and marginal probabilities
    joint_probs = {k: v / total_transitions for k, v in transitions.items()}
    marginal_probs = Counter(macrostates[:-1])
    marginal_probs = {k: v / (len(macrostates) - 1) for k, v in marginal_probs.items()}

    # Calculate conditional entropy
    cond_entropy = sum(
        joint_probs[(i, j)] * math.log2(marginal_probs[i] / joint_probs[(i, j)])
        for (i, j) in joint_probs
    )
    return cond_entropy

# Mersenne Twister test
def mersenne_twister_test(sample_size, macrostate_bins):
    np.random.seed(42)
    sequence = np.random.random(sample_size)
    return conditional_entropy(sequence, macrostate_bins)

# Lagged Fibonacci Generator implementation
class LaggedFibonacciGenerator:
    def __init__(self, p=17, q=5, modulus=2**31, seed=42):
        np.random.seed(seed)
        self.p = p
        self.q = q
        self.modulus = modulus
        self.buffer = [np.random.randint(0, modulus) for _ in range(p)]

    def next(self):
        new_value = (self.buffer[-self.p] + self.buffer[-self.q]) % self.modulus
        self.buffer.pop(0)
        self.buffer.append(new_value)
        return new_value / self.modulus

    def generate_sequence(self, size):
        return [self.next() for _ in range(size)]

# Lagged Fibonacci test
def lagged_fibonacci_test(sample_size, macrostate_bins):
    lfg = LaggedFibonacciGenerator()
    sequence = lfg.generate_sequence(sample_size)
    return conditional_entropy(sequence, macrostate_bins)

# Quantum Random Number Generator (QRNG) implementation
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

    def generate_sequence(self, size):
        return [self.get_random_float() for _ in range(size)]

# QRNG test
def qrng_test(sample_size, macrostate_bins, qrng):
    sequence = qrng.generate_sequence(sample_size)
    return conditional_entropy(sequence, macrostate_bins)

# Compare all PRNGs
if __name__ == "__main__":
    sample_size = 100000  # Size of the random number sequence
    macrostate_values = [16, 32, 64]  # Different macrostate bins
    iterations = 100  # Number of repetitions

    results = {macrostate: {"QRNG > MT": 0, "MT > QRNG": 0} for macrostate in macrostate_values}
    #qrng = RandomBitReader("./random/")

    for macrostate_bins in macrostate_values:
        qrng = RandomBitReader("./random/")
        for _ in tqdm.tqdm(range(iterations), desc=f"Macrostate {macrostate_bins}"):
            mt_entropy = mersenne_twister_test(sample_size, macrostate_bins)
            qrng_entropy = qrng_test(sample_size, macrostate_bins, qrng)

            if qrng_entropy > mt_entropy:
                results[macrostate_bins]["QRNG > MT"] += 1
            else:
                results[macrostate_bins]["MT > QRNG"] += 1

    # Print results
    for macrostate_bins in macrostate_values:
        print(f"Macrostate Bins: {macrostate_bins}")
        print(f"  QRNG > MT: {results[macrostate_bins]['QRNG > MT']}")
        print(f"  MT > QRNG: {results[macrostate_bins]['MT > QRNG']}")

    # Plot results
    labels = [f"{bins}" for bins in macrostate_values]
    qrng_wins = [results[bins]["QRNG > MT"] for bins in macrostate_values]
    mt_wins = [-results[bins]["MT > QRNG"] for bins in macrostate_values]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    # Plot QRNG (upward solid black bars)
    rects1 = ax.bar(x, qrng_wins, width, label='QRNG > MT', color='black', edgecolor='black')

    # Plot MT (downward hollow bars)
    rects2 = ax.bar(x, mt_wins, width, label='MT > QRNG', facecolor='none', edgecolor='blue', linestyle='--')

    # Labeling the plot
    ax.set_xlabel('Macrostate Bins')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Conditional Entropy Wins')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color='black', linewidth=0.8)  # Add a horizontal line at y=0 for clarity
    # ax.legend()


    plt.savefig('entropy2.png', transparent=True)
    plt.show()
    