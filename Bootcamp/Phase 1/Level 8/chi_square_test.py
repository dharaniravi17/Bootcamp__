import scipy.stats as stats

# Given values
observed = [121, 288, 91]  # Observed counts in 2010
expected = [100, 150, 250]  # Expected counts from 2000 distribution

# Perform chi-square test
chi2_stat, p_value = stats.chisquare(observed, expected)

# Decision: Compare with alpha (0.05)
decision = "Reject H₀: Age distribution has changed" if p_value < 0.05 else "Fail to Reject H₀"

# Print results
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-value: {p_value:.4f}")
print(f"Decision: {decision}")
