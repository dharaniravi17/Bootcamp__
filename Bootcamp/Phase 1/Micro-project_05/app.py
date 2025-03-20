import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset (Ensure "house_prices.csv" exists in the working directory)
df = pd.read_csv("C:\\Users\\Dharani Ravi\\AppData\\Local\\Temp\\MicrosoftEdgeDownloads\\7d7163d4-65d1-4de9-9529-dc07a3464f67\\archive\\USA Housing Dataset.csv")

# Step 2: Display summary statistics for price and square footage
print("Summary Statistics:")
print(df[['price', 'sqft_lot']].describe())

# Step 3: Create a new feature (Price per Square Foot)
df['price'] = df['sqft_lot'] / df['sqft_living']

# Step 4: Display the first 5 rows of the updated DataFrame
print("\nUpdated DataFrame:")
print(df.head())

# Step 5: Scatter plot (Price vs. Square Footage)
plt.figure(figsize=(8, 5))
plt.scatter(df['sqft_lot'], df['price'], alpha=0.5, color='blue')
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("House Price vs. Square Footage")
plt.grid(True)

# Step 6: Show the plot
plt.show()
