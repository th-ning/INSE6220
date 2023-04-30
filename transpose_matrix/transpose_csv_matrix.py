import pandas as pd

input_file_path = "project.csv"
df = pd.read_csv(input_file_path, index_col=0)

df_transposed = df.T

output_file_path = "project_transposed.csv"
df_transposed.to_csv(output_file_path)
