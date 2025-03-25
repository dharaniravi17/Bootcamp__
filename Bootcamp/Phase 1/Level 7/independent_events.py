import numpy as np

# Number of trials
num_trials = 1000

# Simulate 1000 pairs of die rolls (random integers from 1 to 6)
die_rolls_1 = np.random.randint(1, 7, num_trials)  # First roll
die_rolls_2 = np.random.randint(1, 7, num_trials)  # Second roll

# Count occurrences where first roll = 5 and second roll = 4
success_count = np.sum((die_rolls_1 == 5) & (die_rolls_2 == 4))

# Compute simulated probability
simulated_prob = success_count / num_trials

# Theoretical probability: P(5) * P(4) = (1/6) * (1/6) = 1/36
theoretical_prob = 1 / 36

# Print results
print(f"### Probability of Rolling a 5 Followed by a 4 ###\n")
print(f"Simulated Probability: {simulated_prob:.4f}")
print(f"Theoretical Probability: {theoretical_prob:.4f} (1/36 ≈ 0.0278)")

print("\nThe simulated probability should be close to the theoretical value.")
