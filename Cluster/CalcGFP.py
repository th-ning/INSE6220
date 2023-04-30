import pandas as pd
import numpy as np

def calculate_gfp(row):
    mean = np.mean(row)
    deviation = row - mean
    squared_deviation = np.square(deviation)
    gfp = np.sqrt(np.mean(squared_deviation))
    return gfp

def process_eeg_data(input_file, output_file):
    # Read the input CSV file
    eeg_data = pd.read_csv(input_file, index_col=0)

    # Calculate the GFP for each row and add it as a new column
    eeg_data['GFP'] = eeg_data.apply(calculate_gfp, axis=1)

    # Write the modified data to a new CSV file
    eeg_data.to_csv(output_file)

def main():
    input_file1 = 'P01_practiceA_1_all_peaks.csv'
    output_file1 = 'Tianhao_gfp.csv'

    input_file2 = 'P01_practiceA_1_peaks.csv'
    output_file2 = 'Mengting_gfp.csv'

    process_eeg_data(input_file1, output_file1)
    process_eeg_data(input_file2, output_file2)

if __name__ == '__main__':
    main()
