#Importing the necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load actual dataset
df = pd.read_csv("rolling_stones_spotify.csv")

# Drop irrelevant or non-numeric columns if they exist
drop_cols = ['name', 'album', 'release_date', 'id', 'uri']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Drop missing values
df = df.dropna()

# Convert all column names to lowercase for consistency
df.columns = [col.lower() for col in df.columns]

# Select only numeric columns for clustering
df_numeric = df.select_dtypes(include=[np.number])

# Check if df_numeric is empty or malformed
if df_numeric.empty:
    raise ValueError("No numeric columns available after filtering. Check your data.")
	
# Remove outliers using Z-score
from scipy.stats import zscore
z_scores = np.abs(zscore(df_numeric))
df_numeric = df_numeric[(z_scores < 3).all(axis=1)]

# Drop columns with object type (extra precaution)
df_numeric = df_numeric.loc[:, df_numeric.dtypes != 'object']

# Final check before scaling
if df_numeric.shape[1] == 0:
    raise ValueError("No valid numeric columns to scale. Check your dataset.")
	
# Normalize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Elbow Method to find optimal clusters
wcss = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_pca)
    wcss.append(kmeans.inertia_)
	
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Silhouette Scores to validate clustering
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca)
    score = silhouette_score(df_pca, labels)
    print(f"Silhouette Score for k={k}: {score:.2f}")
	
# Apply KMeans with optimal number of clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_pca)

# Add cluster labels to original (non-normalized) df
df_result = df_numeric.copy()
df_result['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(8, 5))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title('Song Clusters (KMeans)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')
plt.show()

# Show cluster-wise mean feature values
cluster_summary = df_result.groupby('Cluster').mean(numeric_only=True)
print(cluster_summary)

# Export clustered data
df_result.to_csv("clustered_songs.csv", index=False)
 

