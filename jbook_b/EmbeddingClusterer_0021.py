import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
from typing import List, Dict, Union, Tuple

class EmbeddingClusterer:
    """
    A utility class for clustering embeddings using various algorithms
    and determining the optimal number of clusters. Optimized for embeddings
    where angular distance (cosine similarity) is the appropriate metric.
    """
    
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize the clusterer with embedding vectors.
        
        Args:
            embeddings: numpy array of shape (n_samples, n_dimensions)
        """
        self.embeddings = np.array(embeddings)
        self.normalized_embeddings = self._normalize_embeddings()
        # Pre-compute the angular distance matrix for DBSCAN
        self.angular_distance_matrix = self._compute_angular_distance_matrix()
        
    def _normalize_embeddings(self) -> np.ndarray:
        """Normalize the embeddings to unit length."""
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        return self.embeddings / norms
    
    def _compute_angular_distance_matrix(self) -> np.ndarray:
        """
        Compute pairwise angular distances between all embeddings.
        Angular distance = arccos(cosine_similarity) / π
        Returns normalized distances in [0, 1] range with zero diagonal.
        """
        # Compute cosine similarity matrix
        cos_sim = np.dot(self.normalized_embeddings, self.normalized_embeddings.T)
        # Clip to avoid numerical errors
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        # Convert to angular distance normalized to [0, 1]
        distances = np.arccos(cos_sim) / np.pi
        # Set diagonal to 0 as required for silhouette score
        np.fill_diagonal(distances, 0)
        return distances
        
    def find_optimal_clusters(self, max_clusters: int = 10) -> Tuple[int, float]:
        """
        Find the optimal number of clusters using silhouette analysis.
        Uses angular distance metric for clustering quality assessment.
        
        Args:
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Tuple of (optimal_n_clusters, best_silhouette_score)
        """
        best_score = -1
        optimal_clusters = 2
        
        for n_clusters in range(2, min(max_clusters + 1, len(self.embeddings))):
            # Use spherical k-means approach
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.normalized_embeddings)
            
            if len(np.unique(cluster_labels)) > 1:
                # Use precomputed angular distances for silhouette score
                silhouette_avg = silhouette_score(
                    self.angular_distance_matrix,
                    cluster_labels,
                    metric='precomputed'
                )
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    optimal_clusters = n_clusters
                    
        return optimal_clusters, best_score
    
    def cluster_kmeans(self, n_clusters: int = None) -> Dict[str, Union[np.ndarray, List[int]]]:
        """
        Perform spherical K-means clustering on the embeddings.
        This implementation uses normalized vectors and effectively clusters
        based on cosine similarity.
        
        Args:
            n_clusters: Number of clusters (if None, finds optimal number)
            
        Returns:
            Dictionary containing cluster labels and centroids
        """
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters()
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.normalized_embeddings)
        
        # Normalize the centroids to maintain unit vectors
        centroids = kmeans.cluster_centers_
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        
        return {
            'labels': labels,
            'centroids': centroids
        }
    
    def cluster_dbscan(self, eps: float = 0.3, min_samples: int = 5) -> Dict[str, List[int]]:
        """
        Perform DBSCAN clustering on the embeddings using angular distance.
        
        Args:
            eps: The maximum angular distance between two samples for them to be considered neighbors
                (normalized to [0, 1] range, where 0 means identical direction and 1 means opposite direction)
            min_samples: The minimum number of samples in a neighborhood for a point to be considered a core point
            
        Returns:
            Dictionary containing cluster labels
        """
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'
        )
        labels = dbscan.fit_predict(self.angular_distance_matrix)
        
        return {
            'labels': labels
        }
    
    def reduce_dimensions(self, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensionality of embeddings for visualization using UMAP.
        Configured to preserve angular distances between vectors.
        
        Args:
            n_components: Number of dimensions to reduce to
            
        Returns:
            Reduced dimension embeddings
        """
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            metric='cosine'  # Use cosine distance for UMAP
        )
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