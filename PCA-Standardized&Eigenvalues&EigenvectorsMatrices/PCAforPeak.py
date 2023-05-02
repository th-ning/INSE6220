import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

source_folder = "C:\\Users\\123456\\Documents\\PCA project\\PeaksGFP(StandardAll)"
target_eigenvalues_folder = "C:\\Users\\123456\\Documents\\PCA project\\ReduceDimensionValue(PeaksGFP(StandardAll))"
target_eigenvectors_folder = "C:\\Users\\123456\\Documents\\PCA project\\VectorAfterReduce(PeaksGFP(StandardAll))"
target_final_matrix_folder = "C:\\Users\\123456\\Documents\\PCA project\\FinalMatrix(PeaksGFP(StandardAll))"

# 创建目标文件夹，如果不存在
os.makedirs(target_eigenvalues_folder, exist_ok=True)
os.makedirs(target_eigenvectors_folder, exist_ok=True)
os.makedirs(target_final_matrix_folder, exist_ok=True)

def explained_variance_ratio(eigenvalues):
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    return explained_variance_ratio

for folder in range(1, 25):
    folder_name = f"P{folder:02d}"
    current_folder = os.path.join(source_folder, folder_name)
    target_eigenvalues_subfolder = os.path.join(target_eigenvalues_folder, folder_name)
    target_eigenvectors_subfolder = os.path.join(target_eigenvectors_folder, folder_name)
    target_final_matrix_subfolder = os.path.join(target_final_matrix_folder, folder_name)

    os.makedirs(target_eigenvalues_subfolder, exist_ok=True)
    os.makedirs(target_eigenvectors_subfolder, exist_ok=True)
    os.makedirs(target_final_matrix_subfolder, exist_ok=True)

    for file in os.listdir(current_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(current_folder, file)
            df = pd.read_csv(file_path, index_col=0)

            # Apply PCA
            pca = PCA()
            Z = pca.fit_transform(df)

            # Keep 99% of the explained variance
            eigenvalues = pca.explained_variance_
            eigenvectors = pca.components_
            cumulative_explained_variance = np.cumsum(explained_variance_ratio(eigenvalues))
            n_components = np.argmax(cumulative_explained_variance >= 0.98) + 1

            reduced_eigenvalues = eigenvalues[:n_components]
            reduced_eigenvectors = eigenvectors[:n_components]

            # Save reduced eigenvalues and eigenvectors
            reduced_eigenvalues_df = pd.DataFrame(reduced_eigenvalues, columns=['Eigenvalue'])
            reduced_eigenvectors_df = pd.DataFrame(reduced_eigenvectors, columns=['Eigenvector_{}'.format(i + 1) for i in range(reduced_eigenvectors.shape[1])])

            reduced_eigenvalues_file_path = os.path.join(target_eigenvalues_subfolder, f"{os.path.splitext(file)[0]}_reduced_eigenvalues.csv")
            reduced_eigenvectors_file_path = os.path.join(target_eigenvectors_subfolder, f"{os.path.splitext(file)[0]}_reduced_eigenvectors.csv")

            reduced_eigenvalues_df.to_csv(reduced_eigenvalues_file_path)
            reduced_eigenvectors_df.to_csv(reduced_eigenvectors_file_path)

            # Compute the final matrix
            final_matrix = np.dot(df.values, reduced_eigenvectors.T)


            # Save the final matrix
            final_matrix_df = pd.DataFrame(final_matrix, columns=['PC{}'.format(i + 1) for i in range(n_components)])
            final_matrix_file_path = os.path.join(target_final_matrix_subfolder,
                                                  f"{os.path.splitext(file)[0]}_final_matrix.csv")
            final_matrix_df.to_csv(final_matrix_file_path, index=True)