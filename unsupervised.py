"""
Unsupervised Learning for ASL Hand Gesture Recognition
Implementation of KMeans clustering following lab exercises
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load the cleaned dataset
print("="*60)
print("Loading cleaned dataset...")
print("="*60)
data = pd.read_csv('hand_landmarks_valid.csv')

# Separate features and labels (keep labels to check cluster contents)
X = data.drop(['gesture'], axis=1).to_numpy()
y_true = data['gesture'].to_numpy()

print(f"Dataset shape: {X.shape}")
print(f"Number of instances: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"\nClass distribution:")
print(pd.Series(y_true).value_counts().sort_index())
print()


def plot_clusters_2d(X, cluster_labels, k, feature1=0, feature2=1):
    """
    Plot clusters in 2D using two selected features
    Similar to lab exercise scatter plot
    
    Parameters:
    -----------
    X : array-like
        Feature data
    cluster_labels : array-like
        Cluster assignments from KMeans
    k : int
        Number of clusters
    feature1, feature2 : int
        Indices of features to plot
    """
    plt.figure(figsize=(8, 6))
    
    # Plot each cluster with a different color
    for i in range(k):
        cluster_points = X[cluster_labels == i]
        plt.scatter(cluster_points[:, feature1], cluster_points[:, feature2], 
                   label=f'Cluster {i}', alpha=0.6, edgecolors='k', linewidth=0.5)
    
    plt.xlabel(f'Feature {feature1}', fontsize=11)
    plt.ylabel(f'Feature {feature2}', fontsize=11)
    plt.title(f'KMeans Clustering (K={k})', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'kmeans_clusters_k{k}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: kmeans_clusters_k{k}.png")
    plt.show()


def analyze_cluster_contents(cluster_labels, true_labels):
    """
    Check what true gestures are in each cluster
    Similar to optional lab task: "do you see similar digits in each cluster?"
    
    Parameters:
    -----------
    cluster_labels : array-like
        Cluster assignments from KMeans
    true_labels : array-like
        Actual gesture labels (A-J)
    """
    # Create a DataFrame for easier analysis
    df = pd.DataFrame({
        'cluster': cluster_labels,
        'gesture': true_labels
    })
    
    # Show composition of each cluster
    composition = pd.crosstab(df['cluster'], df['gesture'])
    
    print("\nCluster Contents (rows=clusters, columns=gestures):")
    print(composition)
    print()
    
    # Show which gesture is most common in each cluster
    print("Most common gesture in each cluster:")
    print("-"*60)
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        most_common_gesture = cluster_data['gesture'].mode()[0]
        most_common_count = (cluster_data['gesture'] == most_common_gesture).sum()
        
        print(f"Cluster {cluster_id}: Size={cluster_size:4d}, "
              f"Most common gesture={most_common_gesture} "
              f"({most_common_count}/{cluster_size} instances)")
    print("-"*60)
    print()
    
    return composition


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("Trying Different K Values")
    print("="*60 + "\n")
    
    # Try different K values, similar to lab (k=2, k=4, etc.)
    k_values = [2, 4, 6, 8, 10, 12]
    
    for k in k_values:
        print(f"\n--- KMeans with K={k} ---")
        
        # Fit KMeans - following lab parameters
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        cluster_labels = kmeans.fit_predict(X)
        
        # Print cluster labels (like lab example)
        print(f"Cluster assignments: {cluster_labels}")
        print(f"Number of instances in each cluster:")
        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} instances")
        
        # Plot the clusters (using first two features for visualization)
        plot_clusters_2d(X, cluster_labels, k, feature1=0, feature2=1)
    
    print("\n" + "="*60)
    print("Detailed Analysis for K=10 (same as number of gestures)")
    print("="*60 + "\n")
    
    # Focus on K=10 since we have 10 gesture classes (A-J)
    k_focus = 10
    kmeans_10 = KMeans(n_clusters=k_focus, random_state=SEED, n_init=20)
    cluster_labels_10 = kmeans_10.fit_predict(X)
    
    print(f"KMeans with K={k_focus}:")
    print(f"Cluster assignments for first 20 instances: {cluster_labels_10[:20]}")
    print()
    
    # Check what's in each cluster compared to true labels
    composition = analyze_cluster_contents(cluster_labels_10, y_true)
    
    # Plot with different feature pairs
    plot_clusters_2d(X, cluster_labels_10, k_focus, feature1=0, feature2=1)
    plot_clusters_2d(X, cluster_labels_10, k_focus, feature1=3, feature2=6)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal instances: {X.shape[0]}")
    print(f"Features per instance: {X.shape[1]}")
    print(f"K values tried: {k_values}")
    print(f"\nClustering complete - check saved plots.")
    print("="*60)
