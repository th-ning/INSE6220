import csv
import numpy as np
import matplotlib.pyplot as plt

# Read the EEG data from the CSV file and store it in a list
eeg_data = []
with open("P01_practiceA_1_all.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the first row (header)
    for row in reader:
        eeg_data.append([float(value) for value in row])

# Convert the list to a NumPy array
eeg_data = np.array(eeg_data)

# Calculate the GFP values
def compute_gfp(eeg_data):
    """
    Compute the Global Field Power (GFP) for each time point in the given EEG data.
    The input is a 2D NumPy array where each row represents a time point and each
    column represents an electrode. The output is a 1D NumPy array of GFP values.
    """
    return np.std(eeg_data, axis=1)

gfp_values = compute_gfp(eeg_data)

# Plot the GFP trend
def plot_gfp_trend(gfp_values):
    plt.plot(range(1, len(gfp_values) + 1), gfp_values)
    plt.xlabel("Time Point")
    plt.ylabel("GFP Value")
    plt.title("GFP Trend")
    plt.grid(True)
    plt.show()

plot_gfp_trend(gfp_values)
