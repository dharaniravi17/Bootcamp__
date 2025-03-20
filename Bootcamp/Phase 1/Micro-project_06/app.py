import threading
import logging
import pandas as pd

# Step 1: Configure logging
logging.basicConfig(filename="processor.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Step 2: Define a function to read a CSV and compute the mean
def process_csv(file_path, column_name):
    logging.info(f"Processing started for {file_path}")
    df = pd.read_csv(file_path)

    if column_name in df.columns:
        mean_value = df[column_name].mean()
        logging.info(f"Processing completed for {file_path}, Mean: {mean_value}")
        print(f"Mean of {column_name} in {file_path}: {mean_value:.2f}")
    else:
        logging.error(f"Column '{column_name}' not found in {file_path}")

# Step 3: Create threads for parallel execution
file1 = "C:\\Users\\Dharani Ravi\\Desktop\\ML projects\\stockprice\\env\\Lib\\site-packages\\sklearn\\datasets\\data\\wine_data.csv"
file2 = "C:\\Users\\Dharani Ravi\\Desktop\\ML projects\\stockprice\\env\\Lib\\site-packages\\sklearn\\datasets\\data\\iris.csv"
column_name = "value"  # Replace with the actual numeric column

thread1 = threading.Thread(target=process_csv, args=(file1, column_name))
thread2 = threading.Thread(target=process_csv, args=(file2, column_name))

# Step 4: Start both threads
thread1.start()
thread2.start()

# Step 5: Wait for both threads to finish
thread1.join()
thread2.join()

# Step 6: Indicate completion
print("Processing complete! Check processor.log for details.")
