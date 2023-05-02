import numpy as np
import pandas as pd

def matrix_multiply_csv(file1, file2, output_file):
    # Read CSV files
    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)

    # Remove index row and index column
    df1_values = df1.iloc[1:, 1:].values
    df2_values = df2.iloc[1:, 1:].values

    # Transpose the second matrix
    df2_values_transposed = df2_values.T

    # Ensure the matrices can be multiplied (rows and columns rule)
    if df1_values.shape[1] != df2_values_transposed.shape[0]:
        raise ValueError(f"Matrices cannot be multiplied because the number of columns in the first matrix ({df1_values.shape[1]}) is not equal to the number of rows in the second matrix ({df2_values_transposed.shape[0]})")
    else:
        print((
                  f"Matrices can be multiplied, the number of columns in the first matrix ({df1_values.shape[1]}) is equal to the number of rows in the second matrix ({df2_values_transposed.shape[0]})"))
    # Calculate matrix product
    product = np.dot(df1_values, df2_values_transposed)

    # Create new index and column names
    new_index = range(df1_values.shape[0])
    new_columns = [f"PC{i+1}" for i in range(df2_values_transposed.shape[1])]

    # Save the result to a new DataFrame
    result = pd.DataFrame(product, index=new_index, columns=new_columns)

    # Save the result to a CSV file
    result.to_csv(output_file, index=True, header=True)

# File names to be processed
file1 = 'P01_Merged_All_centered_data.csv'
file2 = 'P01_Merged_All_centered_data_peaks_reduced_eigenvectors.csv'
output_file = 'matrix_product.csv'

# Perform matrix multiplication and save the result to a CSV file
try:
    matrix_multiply_csv(file1, file2, output_file)
    print("Matrix multiplication succeeded! Result saved to:", output_file)
except ValueError as e:
    print("Error:", e)
