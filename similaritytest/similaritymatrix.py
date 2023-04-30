import pandas as pd
from itertools import permutations
from sklearn.metrics.pairwise import cosine_similarity

# Read the three CSV files
df1 = pd.read_csv('P01_All_GFP_Classification.csv', index_col=0)
df2 = pd.read_csv('P01_All_GFP_PCA_Classification.csv', index_col=0)
df3 = pd.read_csv('P01_All_PCA_GFP_Classification.csv', index_col=0)

# Extract the last column (cluster labels)
cluster1 = df1.iloc[:, -1]
cluster2 = df2.iloc[:, -1]
cluster3 = df3.iloc[:, -1]

# Calculate the similarity matrix
similarity_matrix = pd.DataFrame(columns=['P01_All_GFP', 'P01_All_GFP_PCA', 'P01_All_PCA_GFP'],
                                 index=['P01_All_GFP', 'P01_All_GFP_PCA', 'P01_All_PCA_GFP'])

# Calculate similarity between files 1 and 2
max_similarity = 0
for perm in permutations([0, 1, 2]):
    perm_cluster2 = cluster2.map(dict(zip(perm, [0, 1, 2])))
    similarity = cosine_similarity([cluster1], [perm_cluster2])[0][0]
    if similarity > max_similarity:
        max_similarity = similarity
similarity_matrix.loc['P01_All_GFP', 'P01_All_GFP_PCA'] = max_similarity

# Calculate similarity between files 1 and 3
max_similarity = 0
for perm in permutations([0, 1, 2]):
    perm_cluster3 = cluster3.map(dict(zip(perm, [0, 1, 2])))
    similarity = cosine_similarity([cluster1], [perm_cluster3])[0][0]
    if similarity > max_similarity:
        max_similarity = similarity
similarity_matrix.loc['P01_All_GFP', 'P01_All_PCA_GFP'] = max_similarity

# Calculate similarity between files 2 and 3
max_similarity = 0
for perm in permutations([0, 1, 2]):
    perm_cluster3 = cluster3.map(dict(zip(perm, [0, 1, 2])))
    similarity = cosine_similarity([cluster2], [perm_cluster3])[0][0]
    if similarity > max_similarity:
        max_similarity = similarity
similarity_matrix.loc['P01_All_GFP_PCA', 'P01_All_PCA_GFP'] = max_similarity

# Set the diagonal elements to 1, as each file is identical to itself
similarity_matrix.loc['P01_All_GFP', 'P01_All_GFP'] = 1
similarity_matrix.loc['P01_All_GFP_PCA', 'P01_All_GFP_PCA'] = 1
similarity_matrix.loc['P01_All_PCA_GFP', 'P01_All_PCA_GFP'] = 1

# Fill the symmetric positions
similarity_matrix.loc['P01_All_GFP_PCA', 'P01_All_GFP'] = similarity_matrix.loc['P01_All_GFP', 'P01_All_GFP_PCA']
similarity_matrix.loc['P01_All_PCA_GFP', 'P01_All_GFP'] = similarity_matrix.loc['P01_All_GFP', 'P01_All_PCA_GFP']
similarity_matrix.loc['P01_All_PCA_GFP', 'P01_All_GFP_PCA'] = similarity_matrix.loc['P01_All_GFP_PCA', 'P01_All_PCA_GFP']

# Save the similarity matrix to a CSV file
similarity_matrix.to_csv('similarity_matrix.csv')

# Output the number of values, common values, and unique values for each file
values_counts = [len(cluster1), len(cluster2), len(cluster3)]
common_count = len(set(cluster1) & set(cluster2) & set(cluster3))
unique_counts = [len(set(cluster1)), len(set(cluster2)), len(set(cluster3))]

print("File1 values count:", values_counts[0])
print("File2 values count:", values_counts[1])
print("File3 values count:", values_counts[2])
print("Common values count:", common_count)
print("Unique values in File1:", unique_counts[0])
print("Unique values in File2:", unique_counts[1])
print("Unique values in File3:", unique_counts[2])

