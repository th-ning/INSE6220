import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Read the data
    file_path = r"C:\Users\123456\Documents\PCA project\PeaksGFP(StandardAll)\P15\P15_Merged_All_centered_data_peaks.csv"
    data = pd.read_csv(file_path, index_col=0)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(data.cov())

    # Calculate explained variance
    explained_variance = eigenvalues / np.sum(eigenvalues)

    # Plot explained variance
    plot_explained_variance(explained_variance)

    # Plot cumulative explained variance
    plot_cumulative_explained_variance(explained_variance)

def plot_explained_variance(explained_variance):
    x = np.arange(len(explained_variance)) + 1
    plt.plot(x, explained_variance, 'ro-', lw=3)
    plt.xticks(x, [str(i) for i in x], rotation=0)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance Plot')
    plt.grid(True)
    plt.savefig("P15_Explained_Variance_Plot_PeaksGFP(StandardAll).png", dpi=300)
    plt.show()

def plot_cumulative_explained_variance(explained_variance):
    cum_var_exp = np.cumsum(explained_variance)
    x = np.arange(len(cum_var_exp)) + 1

    # Plot bar chart and cumulative line
    fig, ax = plt.subplots()
    ax.bar(x, explained_variance, alpha=0.5)
    ax.plot(x, cum_var_exp, 'ko-', lw=3)

    # Add red vertical line at 98% threshold
    threshold = 0.98
    index_98 = np.argmax(cum_var_exp >= threshold)
    ax.axvline(x=index_98 + 1, color='r', linestyle='--', lw=2)

    # Configure plot settings
    plt.xticks(x, [str(i) for i in x], rotation=0)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Plot')
    plt.grid(True)

    # Save and show plot
    plt.savefig("P15_Cumulative_Explained_Variance_Plot_PeaksGFP(StandardAll).png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
