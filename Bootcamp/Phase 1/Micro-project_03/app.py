import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset from Seaborn
df = sns.load_dataset("titanic")

# Display the first 5 rows
print("\n🔹 First 5 rows of the dataset:")
print(df.head())

# Basic Info
print("\n🔹 Dataset Information:")
print(df.info())

# Summary Statistics
print("\n🔹 Summary Statistics:")
print(df.describe(include="all"))

# Count missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Fill missing values for 'age' with median
df['age'].fillna(df['age'].median(), inplace=True)

# Drop rows where 'embark_town' is missing
df.dropna(subset=['embark_town'], inplace=True)

# Bar plot: Survival Rate by Passenger Class
plt.figure(figsize=(8, 5))
sns.barplot(x="class", y="survived", data=df, ci=None, palette="coolwarm")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.title("Titanic Survival Rate by Class")
plt.show()

# Print survival count by class
survival_counts = df.groupby("class")["survived"].value_counts().unstack()
print("\n🔹 Survival Count by Class:")
print(survival_counts)
