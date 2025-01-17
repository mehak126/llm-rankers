import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
from typing import List, Dict, Union, Tuple

class EmbeddingClusterer:
    """
    A utility class for clustering OpenAI embeddings using various algorithms
    and determining the optimal number of clusters.
    """
    
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize the clusterer with embedding vectors.
        
        Args:
            embeddings: numpy array of shape (n_samples, n_dimensions)
        """
        self.embeddings = np.array(embeddings)
        self.normalized_embeddings = self._normalize_embeddings()
        
    def _normalize_embeddings(self) -> np.ndarray:
        """Normalize the embeddings to unit length."""
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        return self.embeddings / norms
        
    def find_optimal_clusters(self, max_clusters: int = 10) -> Tuple[int, float]:
        """
        Find the optimal number of clusters using silhouette analysis.
        
        Args:
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Tuple of (optimal_n_clusters, best_silhouette_score)
        """
        best_score = -1
        optimal_clusters = 2
        
        for n_clusters in range(2, min(max_clusters + 1, len(self.embeddings))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.normalized_embeddings)
            
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(self.normalized_embeddings, cluster_labels)
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    optimal_clusters = n_clusters
                    
        return optimal_clusters, best_score
    
    def cluster_kmeans(self, n_clusters: int = None) -> Dict[str, Union[np.ndarray, List[int]]]:
        """
        Perform K-means clustering on the embeddings.
        
        Args:
            n_clusters: Number of clusters (if None, finds optimal number)
            
        Returns:
            Dictionary containing cluster labels and centroids
        """
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters()
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.normalized_embeddings)
        
        return {
            'labels': labels,
            'centroids': kmeans.cluster_centers_
        }
    
    def cluster_dbscan(self, eps: float = 0.3, min_samples: int = 5) -> Dict[str, List[int]]:
        """
        Perform DBSCAN clustering on the embeddings.
        
        Args:
            eps: The maximum distance between two samples for them to be considered neighbors
            min_samples: The minimum number of samples in a neighborhood for a point to be considered a core point
            
        Returns:
            Dictionary containing cluster labels
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.normalized_embeddings)
        
        return {
            'labels': labels
        }
    
    def reduce_dimensions(self, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensionality of embeddings for visualization using UMAP.
        
        Args:
            n_components: Number of dimensions to reduce to
            
        Returns:
            Reduced dimension embeddings
        """
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(self.normalized_embeddings)

# Example usage
if __name__ == "__main__":
    # Sample embeddings (replace with your actual embeddings)
    sample_embeddings = np.random.rand(100, 1536)  # OpenAI embeddings are 1536-dimensional
    
    # Initialize clusterer
    clusterer = EmbeddingClusterer(sample_embeddings)
    
    # Find optimal number of clusters
    n_clusters, score = clusterer.find_optimal_clusters()
    print(f"Optimal number of clusters: {n_clusters} (silhouette score: {score:.3f})")
    
    # Perform clustering
    kmeans_results = clusterer.cluster_kmeans()
    dbscan_results = clusterer.cluster_dbscan()
    
    # Reduce dimensions for visualization
    reduced_embeddings = clusterer.reduce_dimensions()