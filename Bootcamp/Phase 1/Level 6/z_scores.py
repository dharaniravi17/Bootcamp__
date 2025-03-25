# Given values
mean = 50
std_dev = 10

# Function to calculate z-score
def calculate_z_score(x, mean, std_dev):
    return (x - mean) / std_dev

# Calculate z-scores
x1 = 60
x2 = 40
z_x1 = calculate_z_score(x1, mean, std_dev)
z_x2 = calculate_z_score(x2, mean, std_dev)

# Print results
print("### Z-Scores and Interpretation ###\n")
print(f"For x = {x1}: Z-score = {z_x1}")
print(f"For x = {x2}: Z-score = {z_x2}")

# Interpretation
print("\n### Interpretation ###")
print(f"A Z-score of {z_x1} means x = {x1} is 1 standard deviation above the mean.")
print(f"A Z-score of {z_x2} means x = {x2} is 1 standard deviation below the mean.")
