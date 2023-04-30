import os
import pandas as pd
from scipy.stats import f_oneway

# Define the directory where the csv files are located
input_dir = "C:\\Users\\123456\\Documents\\PCA project\\SimilarityOfClassification"

# Initialize lists to store the relationship values between elements 1-2 and 1-3
relationship_1_2 = []
relationship_1_3 = []

# Loop through the first ten files in the directory
file_counter = 0
for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and file_counter < 10:
        # Read the csv file and extract the relationship values
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, index_col=0)
        relationship_1_2.append(df.iloc[0, 1])
        relationship_1_3.append(df.iloc[0, 2])
        file_counter += 1

# Perform the one-way ANOVA test on the relationship values
f_stat, p_value = f_oneway(relationship_1_2, relationship_1_3)

# Print the F-statistic and p-value
print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_value:.3f}")
