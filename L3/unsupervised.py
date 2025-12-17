# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# =========================
# 2. Load and Prepare Data
# =========================
df = pd.read_csv('Mall_Customers.csv')

# Select features for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. Apply K-Means (k=5)
# =========================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

print("K-Means Clusters:")
print(df['KMeans_Cluster'].value_counts().sort_index())

# =========================
# 4. Apply DBSCAN
# =========================
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = dbscan_labels

print("\nDBSCAN Clusters (-1 = noise):")
print(df['DBSCAN_Cluster'].value_counts().sort_index())

# =========================
# 5. Plot: Income vs Spending Score
# =========================
plt.figure(figsize=(12, 5))

# K-Means plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(df['Annual Income (k$)'], 
                      df['Spending Score (1-100)'], 
                      c=df['KMeans_Cluster'], 
                      cmap='viridis', 
                      s=50, 
                      alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means: Income vs Spending')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# DBSCAN plot
plt.subplot(1, 2, 2)
unique_clusters = np.unique(dbscan_labels)
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

for i, cluster in enumerate(unique_clusters):
    mask = dbscan_labels == cluster
    if cluster == -1:
        plt.scatter(df.loc[mask, 'Annual Income (k$)'], 
                   df.loc[mask, 'Spending Score (1-100)'], 
                   c='black', s=30, alpha=0.5, label='Noise')
    else:
        plt.scatter(df.loc[mask, 'Annual Income (k$)'], 
                   df.loc[mask, 'Spending Score (1-100)'], 
                   c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {cluster}')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN: Income vs Spending')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =========================
# 6. Plot: Age vs Spending Score
# =========================
plt.figure(figsize=(12, 5))

# K-Means plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(df['Age'], 
                      df['Spending Score (1-100)'], 
                      c=df['KMeans_Cluster'], 
                      cmap='viridis', 
                      s=50, 
                      alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means: Age vs Spending')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# DBSCAN plot
plt.subplot(1, 2, 2)
for i, cluster in enumerate(unique_clusters):
    mask = dbscan_labels == cluster
    if cluster == -1:
        plt.scatter(df.loc[mask, 'Age'], 
                   df.loc[mask, 'Spending Score (1-100)'], 
                   c='black', s=30, alpha=0.5, label='Noise')
    else:
        plt.scatter(df.loc[mask, 'Age'], 
                   df.loc[mask, 'Spending Score (1-100)'], 
                   c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {cluster}')

plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN: Age vs Spending')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =========================
# 7. Quick Cluster Summary
# =========================
print("\n" + "="*40)
print("QUICK CLUSTER SUMMARY")
print("="*40)

print("\nK-Means Cluster Centers (original scale):")
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, 
                           columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
print(centroids_df)

print("\nDBSCAN Summary:")
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")