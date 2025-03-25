from scipy.stats import norm

# Given data
mean = 4
std_dev = 1
x_value = 4.25

# Calculate the Z-score
z_score = (x_value - mean) / std_dev

# Compute the area in the upper tail (percentage of scores above 4.25)
percentage_above = norm.sf(z_score) * 100  # Survival function (1 - CDF)

# Print the result
print(f"### Gaussian Distribution Area ###\n")
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
print(f"Z-score for x = {x_value}: {z_score:.2f}")
print(f"Percentage of scores above {x_value}: {percentage_above:.2f}%")
