"""
PCA Batch Processing Script

This script performs PCA (Principal Component Analysis) on CSV files in a source directory and saves the resulting eigenvalues, eigenvectors, and centered data to target directories. It loops through the files in the source directory, applies PCA to each file, and saves the results to separate CSV files.

Source Folder: The directory containing the input CSV files.
Target Standard Data Folder: The directory to save the centered data.
Target Eigenvalues Folder: The directory to save the calculated eigenvalues.
Target Eigenvectors Folder: The directory to save the calculated eigenvectors.

For each file in the source folder:
1. Read the CSV file.
2. Check for NaN values and handle them as appropriate.
3. Center the data by subtracting the mean.
4. Save the centered data to the target standard data folder.
5. Apply PCA to the centered data.
6. Save the calculated eigenvalues to the target eigenvalues folder.
7. Save the calculated eigenvectors to the target eigenvectors folder.

Please make sure to update the source folder and target directories according to your specific file paths.

The difference between the old and new files lies in the PCA standardization method.
In the old file, the standardization method subtracts the row mean and divides by the standard deviation (resulting in standardized matrix with variance close to 1 and mean of 0).
In the new file, the standardization method subtracts the row mean only (resulting in mean of 0).
We will use the PCA standardization method described in the INSE6220 course for further operations, which is implemented in the new file.
"""


import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




'''source_folder = "C:\\Users\\123456\\Documents\\PCA project\\MergedPeaks"
target_standard_data_folder = "C:\\Users\\123456\\Documents\\PCA project\\StandardDataMatrix(PCA)(Peaks)(new)"
target_eigenvalues_folder = "C:\\Users\\123456\\Documents\\PCA project\\Eigenvalues(Peaks)(new)"
target_eigenvectors_folder = "C:\\Users\\123456\\Documents\\PCA project\\Eigenvectors(Peaks)(new)"'''


source_folder = "C:\\Users\\123456\\Documents\\PCA project\\MergedAll"
target_standard_data_folder = "C:\\Users\\123456\\Documents\\PCA project\\StandardDataMatrix(PCA)(All)(new)"
target_eigenvalues_folder = "C:\\Users\\123456\\Documents\\PCA project\\Eigenvalues(All)(new)"
target_eigenvectors_folder = "C:\\Users\\123456\\Documents\\PCA project\\Eigenvectors(All)(new)"

# 创建目标文件夹，如果不存在
os.makedirs(target_standard_data_folder, exist_ok=True)
os.makedirs(target_eigenvalues_folder, exist_ok=True)
os.makedirs(target_eigenvectors_folder, exist_ok=True)

for folder in range(1, 25):
    folder_name = f"P{folder:02d}"
    current_folder = os.path.join(source_folder, folder_name)
    target_standard_data_subfolder = os.path.join(target_standard_data_folder, folder_name)
    target_eigenvalues_subfolder = os.path.join(target_eigenvalues_folder, folder_name)
    target_eigenvectors_subfolder = os.path.join(target_eigenvectors_folder, folder_name)

    os.makedirs(target_standard_data_subfolder, exist_ok=True)
    os.makedirs(target_eigenvalues_subfolder, exist_ok=True)
    os.makedirs(target_eigenvectors_subfolder, exist_ok=True)

    for file in os.listdir(current_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(current_folder, file)
            df = pd.read_csv(file_path)

            # Check for NaN values
            if df.isnull().values.any():
                print(f"{file_path} contains NaN values!")
                # Handle NaN values here as appropriate

            # Center the data (subtract the mean)
            X_centered = df.subtract(df.mean())

            # Save centered data
            centered_file_path = os.path.join(target_standard_data_subfolder,
                                              f"{os.path.splitext(file)[0]}_centered_data.csv")
            X_centered.to_csv(centered_file_path, index=True)

            # Apply PCA
            pca = PCA()
            Z = pca.fit_transform(X_centered)

            # Save PCA eigenvalues
            eigenvalues = pd.DataFrame(pca.explained_variance_, columns=['Eigenvalue'])
            eigenvalues_file_path = os.path.join(target_eigenvalues_subfolder,
                                                 f"{os.path.splitext(file)[0]}_eigenvalues.csv")
            eigenvalues.to_csv(eigenvalues_file_path, index=True)

            # Save PCA eigenvectors
            eigenvectors = pd.DataFrame(pca.components_, columns=['Eigenvector_{}'.format(i + 1) for i in
                                                                  range(pca.components_.shape[1])])
            eigenvectors_file_path = os.path.join(target_eigenvectors_subfolder,
                                                  f"{os.path.splitext(file)[0]}_eigenvectors.csv")
            eigenvectors.to_csv(eigenvectors_file_path, index=True)
