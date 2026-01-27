import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Any


class KMeansTrainer:
    """Handles training and prediction for K-Means Clustering models."""

    def __init__(self):
        self.model: KMeans | None = None
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] = []

    def train(
        self,
        data: list[dict],
        feature_columns: list[str],
        n_clusters: int = 3,
    ) -> dict[str, Any]:
        """
        Train a K-Means clustering model on the provided data.

        Args:
            data: List of dictionaries containing the dataset
            feature_columns: List of column names to use as features
            n_clusters: Number of clusters to form

        Returns:
            Dictionary containing metrics, centroids, and cluster assignments
        """
        df = pd.DataFrame(data)

        # Validate columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")

        # Prepare features
        X = df[feature_columns].values

        # Handle any NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        original_indices = np.where(mask)[0].tolist()

        if len(X) < 10:
            raise ValueError("Insufficient data after removing NaN values. Need at least 10 rows.")

        # Validate n_clusters
        if n_clusters >= len(X):
            n_clusters = max(2, len(X) // 3)
        if n_clusters < 2:
            n_clusters = 2

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.model.fit_predict(X_scaled)
        self.feature_columns = feature_columns

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels) if n_clusters > 1 and n_clusters < len(X) else 0
        calinski = calinski_harabasz_score(X_scaled, labels) if n_clusters > 1 else 0
        davies = davies_bouldin_score(X_scaled, labels) if n_clusters > 1 else 0

        # Calculate cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_data = X[cluster_mask]
            cluster_stats.append({
                "cluster_id": i,
                "size": int(cluster_mask.sum()),
                "percentage": round(float(cluster_mask.sum() / len(X) * 100), 2),
                "feature_means": {
                    col: round(float(cluster_data[:, j].mean()), 4)
                    for j, col in enumerate(feature_columns)
                },
                "feature_stds": {
                    col: round(float(cluster_data[:, j].std()), 4)
                    for j, col in enumerate(feature_columns)
                },
            })

        metrics = {
            "silhouette_score": round(silhouette, 4),
            "calinski_harabasz_score": round(calinski, 4),
            "davies_bouldin_score": round(davies, 4),
            "inertia": round(float(self.model.inertia_), 4),
            "n_clusters": n_clusters,
            "n_samples": len(X),
            "cluster_stats": cluster_stats,
        }

        # Prepare centroids (unscaled for interpretability)
        centroids_scaled = self.model.cluster_centers_
        centroids_unscaled = self.scaler.inverse_transform(centroids_scaled)
        centroids = [
            {col: round(float(centroids_unscaled[i][j]), 4) for j, col in enumerate(feature_columns)}
            for i in range(n_clusters)
        ]

        # Prepare scaler parameters for predictions
        scaler_params = {
            "mean": {col: float(m) for col, m in zip(feature_columns, self.scaler.mean_)},
            "scale": {col: float(s) for col, s in zip(feature_columns, self.scaler.scale_)},
        }

        # Store scaled centroids for predictions
        centroids_scaled_dict = [
            {col: float(centroids_scaled[i][j]) for j, col in enumerate(feature_columns)}
            for i in range(n_clusters)
        ]

        # Prepare cluster assignments for visualization
        assignments = []
        for i in range(len(X)):
            feature_dict = {col: float(X[i][j]) for j, col in enumerate(feature_columns)}
            assignments.append({
                "cluster": int(labels[i]),
                "features": feature_dict,
                "original_index": original_indices[i],
            })

        # Generate Python code
        generated_code = self._generate_code(feature_columns, n_clusters)

        return {
            "metrics": metrics,
            "centroids": centroids,
            "centroids_scaled": centroids_scaled_dict,
            "scaler_params": scaler_params,
            "assignments": assignments,
            "generated_code": generated_code,
        }

    def predict(
        self,
        data: list[dict],
        feature_columns: list[str],
        centroids_scaled: list[dict],
        scaler_params: dict,
    ) -> list[dict]:
        """
        Assign new data points to clusters.

        Args:
            data: List of dictionaries containing feature data
            feature_columns: List of feature column names
            centroids_scaled: List of scaled centroid dictionaries
            scaler_params: Dictionary with mean and scale for standardization

        Returns:
            List of cluster assignment dictionaries
        """
        df = pd.DataFrame(data)
        predictions = []

        # Convert centroids to numpy array
        centroids = np.array([
            [c[col] for col in feature_columns]
            for c in centroids_scaled
        ])

        for _, row in df.iterrows():
            # Scale features
            scaled_features = []
            for col in feature_columns:
                mean = scaler_params["mean"].get(col, 0)
                scale = scaler_params["scale"].get(col, 1)
                scaled = (row[col] - mean) / scale
                scaled_features.append(scaled)

            X_new = np.array(scaled_features)

            # Find nearest centroid
            distances = np.sqrt(((centroids - X_new) ** 2).sum(axis=1))
            cluster = int(np.argmin(distances))

            predictions.append({
                "cluster": cluster,
                "distances": {i: round(float(d), 4) for i, d in enumerate(distances)},
            })

        return predictions

    def _generate_code(
        self,
        feature_columns: list[str],
        n_clusters: int,
    ) -> str:
        """Generate Python code that replicates the clustering process."""
        features_str = ", ".join(f'"{col}"' for col in feature_columns)

        code = f'''import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("your_data.csv")

# Define features
feature_columns = [{features_str}]

# Prepare the data
X = df[feature_columns]

# Scale features (important for K-Means - distance-based algorithm)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the model
n_clusters = {n_clusters}
model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

# Evaluate the model
silhouette = silhouette_score(X_scaled, labels)
calinski = calinski_harabasz_score(X_scaled, labels)
davies = davies_bouldin_score(X_scaled, labels)

print(f"Number of clusters: {{n_clusters}}")
print(f"Silhouette Score: {{silhouette:.4f}} (higher is better, range: -1 to 1)")
print(f"Calinski-Harabasz Score: {{calinski:.4f}} (higher is better)")
print(f"Davies-Bouldin Score: {{davies:.4f}} (lower is better)")
print(f"Inertia: {{model.inertia_:.4f}}")

# Cluster sizes
for i in range(n_clusters):
    print(f"Cluster {{i}}: {{(labels == i).sum()}} samples")

# Get centroids (unscaled)
centroids = scaler.inverse_transform(model.cluster_centers_)
print(f"\\nCentroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {{i}}: {{dict(zip(feature_columns, centroid.round(4)))}}")

# Optional: Visualize clusters (for 2D data)
if len(feature_columns) == 2:
    plt.scatter(X[feature_columns[0]], X[feature_columns[1]], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.title('K-Means Clustering')
    plt.show()'''

        return code
