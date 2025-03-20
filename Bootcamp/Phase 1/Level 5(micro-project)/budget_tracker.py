import csv
import os
from datetime import datetime

# File to store budget data
BUDGET_FILE = "budget_data.csv"

# Function to save budget data to CSV
def save_to_csv(income, expenses):
    with open(BUDGET_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Category", "Amount", "Date"])

        # Write income data
        for source, amount in income.items():
            writer.writerow(["Income", source, amount, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        # Write expenses data
        for category, amount in expenses.items():
            writer.writerow(["Expense", category, amount, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# Function to load previous budget data from CSV
def load_from_csv():
    income = {}
    expenses = {}

    if not os.path.exists(BUDGET_FILE):
        return income, expenses  # Return empty dictionaries if file doesn't exist

    with open(BUDGET_FILE, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
            if row[0] == "Income":
                income[row[1]] = float(row[2])
            elif row[0] == "Expense":
                expenses[row[1]] = float(row[2])

    return income, expenses

# Function to calculate total income
def calculate_total_income(income_dict):
    return sum(income_dict.values())

# Function to calculate total expenses
def calculate_total_expenses(expenses_dict):
    return sum(expenses_dict.values())

# Function to calculate balance
def calculate_balance(income_total, expenses_total):
    return income_total - expenses_total

# Function to get user input for income and expenses
def get_user_data():
    income, expenses = load_from_csv()

    print("\n🔹 **Existing Budget Data Loaded!**")
    display_summary(income, expenses)

    # Getting new income details
    n = int(input("\nHow many new income sources? (Enter 0 to skip): "))
    for _ in range(n):
        source = input("Enter income source: ")
        amount = float(input(f"Enter amount for {source}: ₹"))
        income[source] = amount

    # Getting new expense details
    m = int(input("\nHow many new expense categories? (Enter 0 to skip): "))
    for _ in range(m):
        category = input("Enter expense category: ")
        amount = float(input(f"Enter amount for {category}: ₹"))
        expenses[category] = amount

    # Save updated data
    save_to_csv(income, expenses)

    return income, expenses

# Function to display budget summary
def display_summary(income, expenses):
    total_income = calculate_total_income(income)
    total_expenses = calculate_total_expenses(expenses)
    balance = calculate_balance(total_income, total_expenses)

    print("\n📊 **Budget Summary**")
    print(f"💰 Total Income: ₹{total_income}")
    print(f"💸 Total Expenses: ₹{total_expenses}")
    print(f"📉 Balance: ₹{balance}")

# Main function
def main():
    income, expenses = get_user_data()
    display_summary(income, expenses)

# Run the budget tracker
if __name__ == "__main__":
    main()
