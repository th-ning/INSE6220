import pandas as pd

def find_nan_locations(csv_path):
    df = pd.read_csv(csv_path)
    nan_locations = df.isnull().stack().reset_index()
    nan_locations.columns = ['Row', 'Column', 'HasNaN']
    nan_locations = nan_locations[nan_locations['HasNaN']]
    return nan_locations

# Example usage
csv_file_path = 'C:\\Users\\123456\\Documents\\PCA project\\MergedAll\\P01\\P01_Merged_All.csv'
nan_locations = find_nan_locations(csv_file_path)
if nan_locations.empty:
    print("No NaN values found in the CSV.")
else:
    print("NaN values found at the following locations:")
    print(nan_locations)
