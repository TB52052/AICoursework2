import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mode

# Configuring the input and output files
INPUT_FILE = "hand_landmarks_valid.csv"
OUTPUT_DIR = "Unsupervised_visualization_output"
SEED = 42  # Setting the seed for reproducibility

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Data loading and preprocessing

print("\n" + "="*40)
print("LOADING & PREPROCESSING")
print("="*40)

try: # checking if the input file exists
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {INPUT_FILE} with shape: {df.shape}")
except FileNotFoundError: # if the input file does not exist, print an error message
    print(f"Error: {INPUT_FILE} not found. Run your feature extraction first.")
    exit()

# Separating the features and metadata
metadata_cols = ['gesture', 'filename', 'hand', 'sample_id', 'Unnamed: 0']
cols_to_drop = [c for c in metadata_cols if c in df.columns]

X_raw = df.drop(columns=cols_to_drop)
y_true = df['gesture'] 

# Dropping the Z-coordinates to reduce noise, this makes the model work work far better
# This is because the z-coordinates are not as important as the x and y coordinates.
z_cols = [col for col in X_raw.columns if '_z' in col]
X_no_z = X_raw.drop(columns=z_cols)
print(f"Dropped {len(z_cols)} Z-axis columns to reduce noise.")

# Scaling the data using MinMax scaling to normalize the data.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_no_z)


# Optimising K using Elbow & Silhouette plot

print("\n" + "="*40)
print(" OPTIMISING K (For K-Means)")
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


# Running the clustering algorithms

k_optimal = 10
print(f"\n" + "="*40)
print(f" RUNNING CLUSTERING MODELS")
print("="*40)

# K-Means, this is the main clustering model we are using. We use k=10 as it is the optimal number of clusters.
# it works by assigning each data point to the cluster with the nearest centroid.
kmeans_final = KMeans(n_clusters=k_optimal, random_state=SEED, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X_scaled)

# Hierarchical, this is a hierarchical clustering model. It works by agglomerating the data points into clusters.
hc = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)


# Plotting and Visualization

print("\nGenerating PCA Visualizations, this is a dimensionality reduction technique that helps us visualize the data in 2D.")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(18, 6))

# Ground Truth, it represents the true labels of the data.
plt.subplot(1, 3, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_true, palette='tab10', s=15, legend=False)
plt.title('Ground Truth')

# K-Means
plt.subplot(1, 3, 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_kmeans, palette='viridis', s=15, legend=False)
plt.title(f'K-Means (k={k_optimal})')

# Hierarchical
plt.subplot(1, 3, 3)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_hc, palette='viridis', s=15, legend=False)
plt.title(f'Hierarchical (k={k_optimal})')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, f"clustering_comparison_all.png")
plt.savefig(plot_path)
print(f"Comparison plot saved to: {plot_path}")


# Evaluation of the clustering models
def evaluate_model(model_name, labels, y_true):
    """
    Prints confusion matrix, ARI, and Accuracy for a given clustering result.
    """
    print("\n" + "-"*60)
    print(f"EVALUATION: {model_name.upper()}")
    print("-"*60)
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(y_true, labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    
    # Cluster Accuracy (Purity)
    df_acc = pd.DataFrame({'True': y_true, 'Cluster': labels})
    total_correct = 0
    total_samples = len(y_true)
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1: continue 
        
        cluster_data = df_acc[df_acc['Cluster'] == cluster_id]
        if len(cluster_data) == 0: continue
            
        most_common_label = cluster_data['True'].mode()[0]
        correct_in_cluster = (cluster_data['True'] == most_common_label).sum()
        total_correct += correct_in_cluster
        
    accuracy = total_correct / total_samples
    print(f"Cluster Purity (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion Matrix (Crosstab)
    print("\nConfusion Matrix (Rows=Cluster, Cols=True Label):")
    cm = pd.crosstab(df_acc['Cluster'], df_acc['True'])
    print(cm)
    
    return ari, accuracy

print("\n" + "="*40)
print("FINAL RESULTS REPORT")
print("="*40)

ari_km, acc_km = evaluate_model("K-Means", labels_kmeans, y_true)
ari_hc, acc_hc = evaluate_model("Hierarchical", labels_hc, y_true)

# Save Dendrogram (Requirement)
plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendrogram")
dendrogram(linkage(X_scaled, method='ward'), truncate_mode='lastp', p=30)
plt.savefig(os.path.join(OUTPUT_DIR, "dendrogram.png"))

# Comparing the models and learnings

print("\n" + "="*40)
print("7. GENERATING COMPARISON GRAPHS")
print("="*40)


# K-Means vs Hierarchical Comparison

def plot_clustering_comparison_graph(ari1, acc1, ari2, acc2):
    """Generates a bar chart comparing ARI and Accuracy for K-Means vs Hierarchical"""
    labels = ['ARI Score', 'Accuracy (Purity)']
    km_vals = [ari1, acc1]
    hc_vals = [ari2, acc2]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, km_vals, width, label='K-Means', color='#3498db')
    rects2 = ax.bar(x + width/2, hc_vals, width, label='Hierarchical', color='#e67e22')

    ax.set_ylabel('Score (0-1)')
    ax.set_title('Clustering Effectiveness: K-Means vs Hierarchical')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison_clusters.png"))
    print("-> Saved: metrics_comparison_clusters.png")

plot_clustering_comparison_graph(ari_km, acc_km, ari_hc, acc_hc)


# Unsupervised vs Supervised Comparison


SUPERVISED_ACCURACY = 0.9919 # This is the accuracy of the best supervised model.

def plot_unsupervised_vs_supervised(unsup_acc, sup_acc):
    """Generates a comparison between best Unsupervised Purity and Supervised Accuracy"""
    labels = ['Unsupervised\n(Best Clustering)', 'Supervised\n(Best Classifier)']
    values = [unsup_acc, sup_acc]
    colors = ['#1abc9c', '#e74c3c']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.6)

    ax.set_ylabel('Performance (Accuracy/Purity)')
    ax.set_title('Unsupervised vs Supervised Performance Gap')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "unsupervised_vs_supervised.png"))
    print("-> Saved: unsupervised_vs_supervised.png")

plot_unsupervised_vs_supervised(max(acc_km, acc_hc), SUPERVISED_ACCURACY)

# Comparison Discussion, this is a discussion of the results of the different learning models.
print("\n" + "-"*60)
print("COMPARISON OF FINDINGS")
print("-" * 60)
print("Clustering Effectiveness:")
if ari_km > ari_hc:
    print(f"   K-Means performed better with an ARI of {ari_km:.4f} compared to Hierarchical ({ari_hc:.4f}).")
    print("   This suggests the dataset has distinct, compact clusters that fit the K-Means centroid model well.")
else:
    print(f"   Hierarchical clustering performed better with an ARI of {ari_hc:.4f}.")

print("\n UNSUPERVISED VS SUPERVISED:")
print(f"   Best Clustering Purity: {max(acc_km, acc_hc):.2%}")
print(f"   Best Supervised Accuracy: {SUPERVISED_ACCURACY:.2%}")
print("\n   DISCUSSION:")
print("   The supervised models significantly outperform clustering. This is expected because:")
print("   a) Unsupervised algorithms lack class labels to distinguish visually similar signs (e.g., A/M/N/S).")
print("   b) They group based on geometric similarity alone, whereas classifiers learn specific decision boundaries.")

