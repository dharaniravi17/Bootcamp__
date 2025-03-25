import numpy as np

# Simulate 1000 Bernoulli trials (P(success) = 0.3)
trials = np.random.binomial(n=1, p=0.3, size=1000)

# Compute simulated success probability
simulated_p = np.mean(trials)

# Print results
print(f"Simulated Probability: {simulated_p:.3f}")
print("Matches theoretical P = 0.3 for Bernoulli trial." if abs(simulated_p - 0.3) < 0.02 else "Deviation from expected value.")
