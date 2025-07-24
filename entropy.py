import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate entropy
def calculate_entropy(data):
    n = len(data)
    if n == 0:
        return 0

    counts = {0: 0, 1: 0}
    for bit in data:
        counts[bit] += 1

    p0 = counts[0] / n
    p1 = counts[1] / n

    entropy = 0
    for p in [p0, p1]:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy

# Generate random bits using Mersenne Twister
mt_rng = random.Random()
mt_rng.seed(42)  # Seed for reproducibility

window_size = 10000  # Number of bits in each window
num_windows = 1000  # Total number of windows

entropy_values = []
time_values = []

for i in range(num_windows):
    bits = [mt_rng.randint(0, 1) for _ in range(window_size)]
    entropy = calculate_entropy(bits)

    entropy_percentage = (entropy / 1.0) * 100  # Convert to percentage (max entropy = 1.0 for binary)
    entropy_values.append(entropy_percentage)
    time_values.append(i)  # Simulated time steps

# Plot the entropy graph
plt.figure(figsize=(10, 6))
plt.plot(time_values, entropy_values, label="Mersenne Twister", color="blue")
plt.title("Information Entropy Over Time")
plt.xlabel("Time (arbitrary units)")
plt.ylabel("Information Entropy (%)")
plt.ylim(99.9, 100.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
