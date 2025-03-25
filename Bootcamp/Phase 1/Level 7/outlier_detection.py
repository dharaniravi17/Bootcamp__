import numpy as np

# Given dataset
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data, ddof=0)  # Population standard deviation

# Compute z-scores
z_scores = [(x - mean) / std_dev for x in data]

# Identify outliers (|z| > 3)
outliers = [data[i] for i in range(len(data)) if abs(z_scores[i]) > 3]

# Print results
print(f"### Outlier Detection Using Z-Score ###\n")
print(f"Data: {data}")
print(f"Mean: {mean:.2f}, Standard Deviation: {std_dev:.2f}")
print(f"Z-Scores: {[round(z, 2) for z in z_scores]}")
print(f"Outliers: {outliers if outliers else 'No outliers detected; all z-scores are below 3.'}")
