import numpy as np
import scipy.stats as stats

# Define height and weight arrays
height = np.array([160, 170, 180, 175])
weight = np.array([60, 70, 75, 65])

# Compute Pearson correlation
pearson_corr, _ = stats.pearsonr(height, weight)

# Compute Spearman correlation
spearman_corr, _ = stats.spearmanr(height, weight)

# Compare results
print(f"Pearson Correlation: {pearson_corr:.2f}")
print(f"Spearman Correlation: {spearman_corr:.2f}")

if pearson_corr > spearman_corr:
    print("Pearson is higher due to a more linear relationship.")
else:
    print("Spearman is higher, indicating a stronger rank-based correlation.")
