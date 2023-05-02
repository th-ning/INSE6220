"""
This script reads multiple CSV files, extracts the matrices from each file, and performs calculations on the matrices.
It calculates the mean, maximum, minimum, and standard deviation matrices and saves the mean matrix to a new CSV file.

The script uses the following libraries:
- os: For file and directory operations
- pandas: For data manipulation and analysis
- numpy: For mathematical operations

Author: Tianhao Ning
"""

import os
import pandas as pd
import numpy as np

# Define the directory where the csv files are located
input_dir = "C:\\Users\\123456\\Documents\\PCA project\\SimilarityOfClassification"
output_dir = "C:\\Users\\123456\\Documents\\PCA project"
output_file = "accuracy.csv"

# Initialize an empty list to store matrices
matrices = []

# Loop through all the files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        # Read the csv file and append its matrix to the list of matrices
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, index_col=0)
        matrix = df.to_numpy()
        matrices.append(matrix)

# Stack matrices along the third dimension
stacked_matrices = np.stack(matrices, axis=2)

# Calculate the mean matrix
mean_matrix = np.mean(stacked_matrices, axis=2)

# Find the maximum and minimum values for each cell
max_matrix = np.max(stacked_matrices, axis=2)
min_matrix = np.min(stacked_matrices, axis=2)

# Calculate the standard deviation for each cell
std_matrix = np.std(stacked_matrices, axis=2)

# Print the maximum, minimum values, and standard deviation
print("Maximum values:")
print(max_matrix)
print("Minimum values:")
print(min_matrix)
print("Standard deviation:")
print(std_matrix)

# Get the first csv file's name and use it as the index name for the output dataframe
first_csv_name = os.listdir(input_dir)[0]
index_name = os.path.splitext(first_csv_name)[0]

# Create a dataframe for the mean matrix with the same index and columns as the original dataframes
mean_df = pd.DataFrame(mean_matrix, index=df.index, columns=df.columns)
mean_df.index.name = index_name

# Save the output dataframe to a csv file in the output directory
mean_df.to_csv(os.path.join(output_dir, output_file))
