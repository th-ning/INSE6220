import os
import pandas as pd
from itertools import permutations
from sklearn.metrics.pairwise import cosine_similarity

# Set the data directories
data_dir1 = r'C:\Users\123456\Documents\PCA project\Classification(StandardAll)'
data_dir2 = r'C:\Users\123456\Documents\PCA project\Classification(All_GFP_PCA)'
data_dir3 = r'C:\Users\123456\Documents\PCA project\Classification(All_PCA_GFP)'

# Set the output directory for similarity matrices
output_dir = r'C:\Users\123456\Documents\PCA project\SimilarityofClassification'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize counters
comparison_count_12 = 0
same_count_12 = 0
different_count_12 = 0
comparison_count_13 = 0
same_count_13 = 0
different_count_13 = 0
comparison_count_23 = 0
same_count_23 = 0
different_count_23 = 0

# Loop through the data files
for i in range(1, 11):
    # Build file paths
    file_path1 = os.path.join(data_dir1, f'P{i:02}', f'P{i:02}_All_GFP_Classification.csv')
    file_path2 = os.path.join(data_dir2, f'P{i:02}', f'P{i:02}_All_GFP_PCA_Classification.csv')
    file_path3 = os.path.join(data_dir3, f'P{i:02}', f'P{i:02}_All_PCA_GFP_Classification.csv')

    # Read the data files
    df1 = pd.read_csv(file_path1, index_col=0)
    df2 = pd.read_csv(file_path2, index_col=0)
    df3 = pd.read_csv(file_path3, index_col=0)

    # Extract the last column (cluster labels)
    cluster1 = df1.iloc[:, -1]
    cluster2 = df2.iloc[:, -1]
    cluster3 = df3.iloc[:, -1]

    # Calculate the similarity matrix
    similarity_matrix = pd.DataFrame(columns=[f'P{i:02}_All_GFP', f'P{i:02}_All_GFP_PCA', f'P{i:02}_All_PCA_GFP'],
                                     index=[f'P{i:02}_All_GFP', f'P{i:02}_All_GFP_PCA', f'P{i:02}_All_PCA_GFP'])

    # Calculate similarity between file 1 and file 2
    max_similarity_12 = 0
    for perm in permutations([0, 1, 2]):
        perm_cluster2 = cluster2.map(dict(zip(perm, [0, 1, 2])))
        similarity = cosine_similarity([cluster1], [perm_cluster2])[0][0]
        if similarity > max_similarity_12:
            max_similarity_12 = similarity
    similarity_matrix.loc[f'P{i:02}_All_GFP', f'P{i:02}_All_GFP_PCA'] = max_similarity_12

    # Calculate similarity between file 1 and file 3
    max_similarity_13 = 0
    for perm in permutations([0, 1, 2]):
        perm_cluster3 = cluster3.map(dict(zip(perm, [0, 1, 2])))
        similarity = cosine_similarity([cluster1], [perm_cluster3])[0][0]
        if similarity > max_similarity_13:
            max_similarity_13 = similarity
    similarity_matrix.loc[f'P{i:02}_All_GFP', f'P{i:02}_All_PCA_GFP'] = max_similarity_13

    # Fill the symmetric positions of the similarity matrix
    similarity_matrix.loc[f'P{i:02}_All_GFP_PCA', f'P{i:02}_All_GFP'] = similarity_matrix.loc[
        f'P{i:02}_All_GFP', f'P{i:02}_All_GFP_PCA']
    similarity_matrix.loc[f'P{i:02}_All_PCA_GFP', f'P{i:02}_All_GFP'] = similarity_matrix.loc[
        f'P{i:02}_All_GFP', f'P{i:02}_All_PCA_GFP']
    similarity_matrix.loc[f'P{i:02}_All_PCA_GFP', f'P{i:02}_All_GFP_PCA'] = similarity_matrix.loc[
        f'P{i:02}_All_GFP_PCA', f'P{i:02}_All_PCA_GFP']

    # Set diagonal elements to 1
    similarity_matrix.loc[f'P{i:02}_All_GFP', f'P{i:02}_All_GFP'] = 1.0
    similarity_matrix.loc[f'P{i:02}_All_GFP_PCA', f'P{i:02}_All_GFP_PCA'] = 1.0
    similarity_matrix.loc[f'P{i:02}_All_PCA_GFP', f'P{i:02}_All_PCA_GFP'] = 1.0

    # Save the similarity matrix to a file
    output_path = os.path.join(output_dir, f'P{i:02}_Classification_Similarity.csv')
    similarity_matrix.to_csv(output_path, index=True)

    # Update comparison and count statistics
    comparison_count_12 += 1
    if max_similarity_12 == 1.0:
        same_count_12 += 1
    else:
        different_count_12 += 1

    comparison_count_13 += 1
    if max_similarity_13 == 1.0:
        same_count_13 += 1
    else:
        different_count_13 += 1

    comparison_count_23 += 1
    if similarity_matrix.loc[f'P{i:02}_All_GFP_PCA', f'P{i:02}_All_PCA_GFP'] == 1.0:
        same_count_23 += 1
    else:
        different_count_23 += 1

    # Print comparison and count statistics
print("Comparison and Count Statistics:")
print(
    f"Data 2 vs Data 1: Comparison Count = {comparison_count_12}, Same Count = {same_count_12}, Different Count = {different_count_12}")
print(
    f"Data 3 vs Data 1: Comparison Count = {comparison_count_13}, Same Count = {same_count_13}, Different Count = {different_count_13}")
print(
    f"Data 2 vs Data 3: Comparison Count = {comparison_count_23}, Same Count = {same_count_23}, Different Count = {different_count_23}")

