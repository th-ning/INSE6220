import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''source_folder = "C:\\Users\\123456\\Documents\\PCA project\\MergedAll"
target_standard_data_folder = "C:\\Users\\123456\\Documents\\PCA project\\StandardDataMatrix(PCA)(All)"
target_eigenvalues_folder = "C:\\Users\\123456\\Documents\\PCA project\\Eigenvalues(All)"
target_eigenvectors_folder = "C:\\Users\\123456\\Documents\\PCA project\\Eigenvectors(All)"'''


source_folder = "C:\\Users\\123456\\Documents\\PCA project\\MergedPeaks"
target_standard_data_folder = "C:\\Users\\123456\\Documents\\PCA project\\StandardDataMatrix(PCA)(Peaks)"
target_eigenvalues_folder = "C:\\Users\\123456\\Documents\\PCA project\\Eigenvalues(Peaks)"
target_eigenvectors_folder = "C:\\Users\\123456\\Documents\\PCA project\\Eigenvectors(Peaks)"


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

            # Standardize the data
            Xs = StandardScaler().fit_transform(df)
            X_standardized = pd.DataFrame(Xs, columns=df.columns)

            # Save standardized data
            standardized_file_path = os.path.join(target_standard_data_subfolder,
                                                  f"{os.path.splitext(file)[0]}_standard_data.csv")
            X_standardized.to_csv(standardized_file_path, index=True)

            # Apply PCA
            pca = PCA()
            Z = pca.fit_transform(X_standardized)

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
