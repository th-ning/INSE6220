import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

source_folder = "C:\\Users\\123456\\Documents\\PCA project\\PeaksGFP"
target_cluster_folder = "C:\\Users\\123456\\Documents\\PCA project\\ClusterPeaks（GFP)"

os.makedirs(target_cluster_folder, exist_ok=True)

def find_optimal_k_elbow(data, max_clusters=20):
    wcss = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    second_derivative = np.diff(wcss, n=2)
    optimal_k = np.argmax(second_derivative) + 3

    return optimal_k, wcss

def plot_data(data, labels=None, title='Data Visualization'):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)

def process_file(file_path):
    data = pd.read_csv(file_path)

    data_no_index = data.drop(data.columns[0], axis=1)

    optimal_k, wcss = find_optimal_k_elbow(data_no_index)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(data_no_index)

    data['cluster'] = kmeans.labels_

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_name = os.path.basename(os.path.dirname(file_path))
    target_folder = os.path.join(target_cluster_folder, folder_name)

    os.makedirs(target_folder, exist_ok=True)

    output_file_path = os.path.join(target_folder, f"{file_name}_clustered.csv")
    data.to_csv(output_file_path, index=False)

    plt.figure()
    plot_data(data_no_index.values, title=f'{file_name} Original Data Visualization')
    plt.savefig(os.path.join(target_folder, f"{file_name}_original_data.png"))

    plt.figure()
    plot_data(data_no_index.values, kmeans.labels_, title=f'{file_name} Clustered Data Visualization')
    plt.savefig(os.path.join(target_folder, f"{file_name}_clustered_data.png"))

    plt.figure()
    plt.plot(range(1, 21), wcss)
    plt.scatter(optimal_k, wcss[optimal_k - 1], marker='o', color='red', label=f'Optimal K: {optimal_k}')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title(f'{file_name} The Elbow Method')
    plt.legend()
    plt.savefig(os.path.join(target_folder, f"{file_name}_elbow_plot.png"))

for folder_name in range(1, 25):
    current_folder = os.path.join(source_folder, f"P{folder_name:02d}")

    for file in os.listdir(current_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(current_folder, file)
            process_file(file_path)
