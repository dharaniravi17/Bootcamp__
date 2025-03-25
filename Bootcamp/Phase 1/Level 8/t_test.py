import numpy as np
import scipy.stats as stats

# Given values
pop_mean = 100  # Population mean IQ
sample_size = 30
sample_std = 20  # Sample standard deviation
alpha = 0.05     # Significance level

# Generate a sample with mean = 140 and SD = 20
np.random.seed(42)  # For reproducibility
sample_data = np.random.normal(140, 20, sample_size)

# Perform one-sample t-test
t_score, p_value = stats.ttest_1samp(sample_data, pop_mean)

# Decision: Compare with alpha
decision = "Reject H₀: Medication increases IQ" if p_value < alpha else "Fail to Reject H₀"

# Print results
print(f"T-score: {t_score:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Decision: {decision}")
