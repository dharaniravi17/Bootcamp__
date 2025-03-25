import numpy as np

# Define height and weight arrays
height = np.array([160, 170, 180, 175])
weight = np.array([60, 70, 75, 65])

# Compute covariance (sample covariance, ddof=1)
cov_matrix = np.cov(height, weight)
cov_value = cov_matrix[0, 1]

# Interpret results
interpretation = "Positive covariance; as height increases, weight tends to increase." if cov_value > 0 else "No positive relationship."

# Print results
print(f"Covariance: {cov_value:.2f}")
print(f"Interpretation: {interpretation}")
