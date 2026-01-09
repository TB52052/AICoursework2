import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mode

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "hand_landmarks_valid.csv"
OUTPUT_DIR = "Visualization_Output_Files"
SEED = 42  # Constant for reproducibility

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. LOAD DATA & PREPROCESSING
# ==========================================
print("\n" + "="*40)
print("1. LOADING & PREPROCESSING")
print("="*40)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {INPUT_FILE} with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Run your feature extraction first.")
    exit()

# A. Separate Features and Metadata
metadata_cols = ['gesture', 'filename', 'hand', 'sample_id', 'Unnamed: 0']
cols_to_drop = [c for c in metadata_cols if c in df.columns]

X_raw = df.drop(columns=cols_to_drop)
y_true = df['gesture']  # Ground Truth (A-J)

# B. Feature Selection (Dropping Z-Coordinates)
z_cols = [col for col in X_raw.columns if '_z' in col]
X_no_z = X_raw.drop(columns=z_cols)
print(f"Dropped {len(z_cols)} Z-axis columns to reduce noise.")

# C. Scaling (MinMax)
print("Applying MinMax Scaling...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_no_z)

# ==========================================
# 2. OPTIMISATION (Elbow & Silhouette)
# ==========================================
print("\n" + "="*40)
print("2. OPTIMISING K (For K-Means)")
print("="*40)

inertia = []
silhouette_scores = []
K_range = range(2, 15)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

# Plot Optimization
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].plot(K_range, inertia, marker='o', color='tab:blue')
ax[0].set_title('Elbow Method (Inertia)')
ax[0].set_xlabel('Number of Clusters (k)')
ax[0].set_ylabel('Inertia')
ax[0].axvline(x=10, color='red', linestyle='--', label='Expected k=10')

ax[1].plot(K_range, silhouette_scores, marker='o', color='tab:orange')
ax[1].set_title('Silhouette Score')
ax[1].set_xlabel('Number of Clusters (k)')
ax[1].set_ylabel('Score')
ax[1].axvline(x=10, color='red', linestyle='--', label='Expected k=10')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "kmeans_optimization.png")
plt.savefig(plot_path)
print(f"Optimization plots saved to: {plot_path}")

# ==========================================
# 3. RUNNING CLUSTERING MODELS
# ==========================================
k_optimal = 10
print(f"\n" + "="*40)
print(f"3. RUNNING CLUSTERING MODELS")
print("="*40)

# A. K-Means
print("Running K-Means...")
kmeans_final = KMeans(n_clusters=k_optimal, random_state=SEED, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X_scaled)

# B. Hierarchical
print("Running Hierarchical Clustering...")
hc = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)

# C. Density-Based (DBSCAN)
print("Running Density-Based Clustering (DBSCAN)...")
# Note: DBSCAN does not use 'k'. It uses epsilon (distance threshold).
# Since we used MinMax (0-1), a small epsilon like 0.2 or 0.3 is usually good.
dbscan = DBSCAN(eps=0.6, min_samples=5)
labels_density = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(labels_density)) - (1 if -1 in labels_density else 0)
print(f"-> DBSCAN found {n_clusters_db} clusters (and identified noise).")

# ==========================================
# 4. VISUALIZATION (PCA)
# ==========================================
print("\nGenerating PCA Visualizations...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(20, 5))

# Plot 1: Ground Truth
plt.subplot(1, 4, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_true, palette='tab10', s=15, legend=False)
plt.title('Ground Truth')

# Plot 2: K-Means
plt.subplot(1, 4, 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_kmeans, palette='viridis', s=15, legend=False)
plt.title(f'K-Means (k={k_optimal})')

# Plot 3: Hierarchical
plt.subplot(1, 4, 3)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_hc, palette='viridis', s=15, legend=False)
plt.title(f'Hierarchical (k={k_optimal})')

# Plot 4: DBSCAN
plt.subplot(1, 4, 4)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_density, palette='viridis', s=15, legend=False)
plt.title(f'DBSCAN (Found {n_clusters_db})')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, f"clustering_comparison_all.png")
plt.savefig(plot_path)
print(f"Comparison plot saved to: {plot_path}")

# ==========================================
# 5. DETAILED EVALUATION FUNCTION
# ==========================================
def evaluate_model(model_name, labels, y_true):
    """
    Prints confusion matrix, ARI, and Accuracy for a given clustering result.
    """
    print("\n" + "-"*60)
    print(f"EVALUATION: {model_name.upper()}")
    print("-"*60)
    
    # 1. Adjusted Rand Index
    ari = adjusted_rand_score(y_true, labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    
    # 2. Cluster Accuracy (Purity)
    # Map each cluster to the most frequent true label it contains
    df_acc = pd.DataFrame({'True': y_true, 'Cluster': labels})
    total_correct = 0
    total_samples = len(y_true)
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1: continue # Skip noise in DBSCAN
        
        cluster_data = df_acc[df_acc['Cluster'] == cluster_id]
        if len(cluster_data) == 0: continue
            
        most_common_label = cluster_data['True'].mode()[0]
        correct_in_cluster = (cluster_data['True'] == most_common_label).sum()
        total_correct += correct_in_cluster
        
    accuracy = total_correct / total_samples
    print(f"Cluster Purity (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 3. Confusion Matrix (Crosstab)
    print("\nConfusion Matrix (Rows=Cluster, Cols=True Label):")
    # We use crosstab because it handles string labels nicely
    cm = pd.crosstab(df_acc['Cluster'], df_acc['True'])
    print(cm)

# ==========================================
# 6. RUN EVALUATION
# ==========================================
print("\n" + "="*40)
print("FINAL RESULTS REPORT")
print("="*40)

evaluate_model("K-Means", labels_kmeans, y_true)
evaluate_model("Hierarchical", labels_hc, y_true)
evaluate_model("Density-Based (DBSCAN)", labels_density, y_true)

# Save Dendrogram (Requirement)
plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendrogram")
dendrogram(linkage(X_scaled, method='ward'), truncate_mode='lastp', p=30)
plt.savefig(os.path.join(OUTPUT_DIR, "dendrogram.png"))

print("\nProcessing Complete.")