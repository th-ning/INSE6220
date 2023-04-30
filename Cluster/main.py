import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def find_optimal_k_elbow(data, max_clusters=20):
    wcss = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # 计算每个点的二阶导数
    second_derivative = np.diff(wcss, n=2)

    # 找到最大二阶导数对应的 K 值
    optimal_k = np.argmax(second_derivative) + 3

    # 绘制 Elbow 图
    plt.plot(range(1, max_clusters + 1), wcss)
    plt.scatter(optimal_k, wcss[optimal_k - 1], marker='o', color='red', label=f'Optimal K: {optimal_k}')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('The Elbow Method')
    plt.legend()
    plt.show()

    return optimal_k


def plot_data(data, labels=None, title='Data Visualization'):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.show()


# 从文件读取数据
data = pd.read_csv('3.csv')

# 准备数据：删除索引列
data_no_index = data.drop(data.columns[0], axis=1)

# 可视化原始数据
plot_data(data_no_index.values, title='Original Data Visualization')

# 寻找最优的 n_clusters 值
optimal_k = find_optimal_k_elbow(data_no_index)

# 在控制台打印最优的 K 值
print(f'The optimal number of clusters (K) is: {optimal_k}')

# 使用最优的 n_clusters 值进行聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data_no_index)

# 可视化聚类结果
plot_data(data_no_index.values, kmeans.labels_, title='Clustered Data Visualization')

# 将聚类标签添加到原始数据
data['cluster'] = kmeans.labels_

# 输出带有聚类标签的 CSV 文件
data.to_csv('clustered_data.csv', index=False)
