"""
This script reads a CSV file, transposes the data, and saves it as a new CSV file.
The transposed feature matrix can be used in the report.

The script uses the following libraries:
- pandas: For data manipulation and analysis

Author: Tianhao Ning
"""

import pandas as pd

input_file_path = "project.csv"
df = pd.read_csv(input_file_path, index_col=0)

df_transposed = df.T

output_file_path = "project_transposed.csv"
df_transposed.to_csv(output_file_path)
