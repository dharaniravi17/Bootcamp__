import statistics

# Given dataset
data = [3, 5, 5, 6, 7, 100]

# Calculate mean
mean_value = sum(data) / len(data)

# Calculate median
median_value = statistics.median(data)

# Calculate mode
mode_value = statistics.mode(data)

# Print results
print("### Measures of Central Tendency ###\n")
print(f"Dataset: {data}")
print(f"Mean: {mean_value:.2f}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")

# Interpretation
print("\n### Interpretation ###")
print("The mean is high due to the outlier (100).")
print("The median (5.5) and mode (5) better represent the central trend.")
