import pandas as pd
import numpy as np

def calculate_gfp(row):
    mean = np.mean(row)
    deviation = row - mean
    squared_deviation = np.square(deviation)
    gfp = np.sqrt(np.mean(squared_deviation))
    return gfp

def find_peak_rows(data, gfp_col):
    peak_rows = []
    index_values = data.index
    for i in range(1, len(index_values) - 1):
        current = index_values[i]
        prev = index_values[i - 1]
        next_ = index_values[i + 1]
        current_gfp = data.loc[current, gfp_col]
        prev_gfp = data.loc[prev, gfp_col]
        next_gfp = data.loc[next_, gfp_col]

        # Check for local maximum
        if current_gfp > prev_gfp and current_gfp > next_gfp:
            peak_rows.append(current)
        # Check for local minimum
        elif current_gfp < prev_gfp and current_gfp < next_gfp:
            peak_rows.append(current)
    return peak_rows




def main():
    input_file = 'P01_practiceA_1_all.csv'
    output_file = 'P01_practiceA_1_all_peaks.csv'

    # Read the input CSV file
    eeg_data = pd.read_csv(input_file, index_col=0)

    # Calculate the GFP for each row and add it as a new column
    eeg_data['GFP'] = eeg_data.apply(calculate_gfp, axis=1)

    # Find the peak rows using the GFP column
    peak_rows = find_peak_rows(eeg_data, 'GFP')

    # Select the peak rows and drop the GFP column
    eeg_data_peaks = eeg_data.loc[peak_rows].drop(columns=['GFP'])

    # Write the peak rows to a new CSV file
    eeg_data_peaks.to_csv(output_file)

if __name__ == '__main__':
    main()
