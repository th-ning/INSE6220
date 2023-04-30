import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

        if current_gfp > prev_gfp and current_gfp > next_gfp:
            peak_rows.append(current)
        elif current_gfp < prev_gfp and current_gfp < next_gfp:
            peak_rows.append(current)
    return peak_rows

def process_eeg_data(input_path, output_path):
    eeg_data = pd.read_csv(input_path, index_col=0)

    eeg_data_no_index = eeg_data.drop(eeg_data.columns[0], axis=1)
    eeg_data_no_index['GFP'] = eeg_data_no_index.apply(calculate_gfp, axis=1)

    peak_rows = find_peak_rows(eeg_data_no_index, 'GFP')

    eeg_data_peaks = eeg_data.loc[peak_rows]
    eeg_data_peaks.to_csv(output_path)

    plot_gfp(eeg_data_no_index, os.path.dirname(output_path), 'GFP_full')
    plot_gfp(eeg_data_no_index.head(100), os.path.dirname(output_path), 'GFP_100')

def plot_gfp(data, output_dir, file_prefix):
    plt.figure()
    plt.plot(data.index, data['GFP'])
    plt.xlabel('Time')
    plt.ylabel('GFP')
    plt.savefig(os.path.join(output_dir, f'{file_prefix}.png'))

def main():
    input_dir = 'C:\\Users\\123456\\Documents\\PCA project\\StandardDataMatrix(PCA)(All)(new)'
    output_dir = 'C:\\Users\\123456\\Documents\\PCA project\\PeaksGFP(StandardAll)'

    for folder in os.listdir(input_dir):
        input_folder = os.path.join(input_dir, folder)
        output_folder = os.path.join(output_dir, folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in os.listdir(input_folder):
            if file.endswith('.csv'):
                input_file = os.path.join(input_folder, file)
                output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_peaks.csv")
                process_eeg_data(input_file, output_file)

if __name__ == '__main__':
    main()
