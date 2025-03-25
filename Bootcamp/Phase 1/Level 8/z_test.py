import numpy as np
import statsmodels.stats.weightstats as st

# Given values
pop_mean = 100  # Population mean IQ
pop_std = 15    # Population standard deviation
sample_size = 30
alpha = 0.05    # Significance level

# Generate a sample with mean = 140 and SD = 15
np.random.seed(42)  # For reproducibility
sample_data = np.random.normal(140, 15, sample_size)

# Perform one-sample z-test
z_score, p_value = st.ztest(sample_data, value=pop_mean, alternative='larger')

# Decision: Compare with alpha
decision = "Reject H₀: Medication affects IQ" if p_value < alpha else "Fail to Reject H₀"

# Print results
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Decision: {decision}")
