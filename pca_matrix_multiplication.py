import numpy as np
import pandas as pd
import os

def matrix_multiply_csv(file1, file2, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # Read CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Print matrix dimensions
        print(f"Matrix 1 ({file1}): {df1.shape}")
        print(f"Matrix 2 ({file2}): {df2.shape}")
    except FileNotFoundError:
        print(f"Error: Cannot find the file {file1} or {file2}")
        return

    # Remove index column
    df1_values = df1.iloc[:, 1:].values
    df2_values = df2.iloc[:, 1:].values

    # Transpose the second matrix
    df2_values_transposed = df2_values.T

    # Ensure the matrices can be multiplied (rows and columns rule)
    if df1_values.shape[1] != df2_values_transposed.shape[0]:
        raise ValueError(f"Matrices cannot be multiplied because the number of columns in the first matrix ({df1_values.shape[1]}) is not equal to the number of rows in the second matrix ({df2_values_transposed.shape[0]})")
    else:
        print(f"Matrices can be multiplied, the number of columns in the first matrix ({df1_values.shape[1]}) is equal to the number of rows in the second matrix ({df2_values_transposed.shape[0]})")
        print(
            f"the number of rows in the first matrix ({df1_values.shape[0]}), the number of columns in the second matrix ({df2_values_transposed.shape[1]})")

    # Calculate matrix product
    product = np.dot(df1_values, df2_values_transposed)

    # Create new index and column names
    new_index = range(df1_values.shape[0])
    new_columns = [f"PC{i+1}" for i in range(df2_values_transposed.shape[1])]

    # Save the result to a new DataFrame
    result = pd.DataFrame(product, index=new_index, columns=new_columns)

    # Generate output file name
    output_file = os.path.join(output_folder, f"{os.path.basename(output_folder)}_Final_StandardAll_Mul_PeaksEigVector.csv")

    # Save the result to a CSV file
    result.to_csv(output_file, index=True, header=True)
    print(f"Matrix multiplication succeeded! Result saved to: {output_file}")

# Set the folder paths
base_folder1 = r'C:\Users\123456\Documents\PCA project\StandardDataMatrix(PCA)(All)(new)'
base_folder2 = r'C:\Users\123456\Documents\PCA project\VectorAfterReduce(PeaksGFP(StandardAll))'
output_base_folder = r'C:\Users\123456\Documents\PCA project\FinalStandardAllMulPeaksEigenvectors'

# Loop through P01 to P24
for i in range(1, 25):
    folder_name = f'P{str(i).zfill(2)}'
    folder1 = os.path.join(base_folder1, folder_name)
    folder2 = os.path.join(base_folder2, folder_name)
    output_folder = os.path.join(output_base_folder, folder_name)

    # Get the CSV file paths in the folders
    csv_file1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.csv')][0]
    csv_file2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.csv')][0]

    # Perform matrix multiplication and save the result to a CSV file
    try:
        matrix_multiply_csv(csv_file1, csv_file2, output_folder)
    except ValueError as e:
        print(f"Error processing {folder_name}: {e}")
