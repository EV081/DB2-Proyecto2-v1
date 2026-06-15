import numpy as np


class KClustering:
    """Agrupa vectores en clusters usando k-means o k-medoids.

    Uso:
        kc = KClustering(n_centroids=5, clustering_algorithm="kmean")
        kc.reset_centroids(dim=13)
        labels, centroids = kc.clusterize(feature_matrix)
    """

    def __init__(self, n_centroids=100, clustering_algorithm="kmean"):
        """Configura número de centroides y algoritmo."""
        self.centroids = []
        self.n_centroids = n_centroids
        self.clustering_algorithm = clustering_algorithm

    def reset_centroids(self, dim=128):
        """Inicializa centroides con valores aleatorios."""
        self.centroids = [np.random.randn(dim).astype(np.float32) for _ in range(self.n_centroids)]

    def euclidean_distance(self, vector_a, vector_b):
        """Distancia euclidiana entre dos vectores."""
        a = np.asarray(vector_a)
        b = np.asarray(vector_b)
        diff = a - b
        return float(np.sqrt(np.sum(diff * diff)))

    def assign_centroid(self, vector):
        """Índice del centroide más cercano al vector."""
        best_dist = float('inf')
        best_idx = 0
        for i, centroid in enumerate(self.centroids):
            dist = self.euclidean_distance(vector, centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _adjust_centroid_kmean(self, data, assignments):
        """Actualiza centroides como promedio de sus puntos."""
        for i in range(self.n_centroids):
            members = []
            for j in range(len(data)):
                if assignments[j] == i:
                    members.append(data[j])
            if members:
                self.centroids[i] = np.mean(members, axis=0)

    def _adjust_centroid_kmedoid(self, data, assignments):
        """Actualiza centroides como el punto más central del cluster."""
        for i in range(self.n_centroids):
            members_idx = []
            for j in range(len(data)):
                if assignments[j] == i:
                    members_idx.append(j)
            if members_idx:
                best_idx = members_idx[0]
                best_cost = float('inf')
                for candidate in members_idx:
                    cost = 0.0
                    for m in members_idx:
                        cost += self.euclidean_distance(data[candidate], data[m])
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = candidate
                self.centroids[i] = data[best_idx].copy()

    def clusterize(self, matrix):
        """Ejecuta el algoritmo de clustering hasta convergencia."""
        data = np.asarray(matrix)
        n_points = len(data)
        self.n_centroids = min(self.n_centroids, n_points)

        for _ in range(100):
            assignments = [-1] * n_points
            for i in range(n_points):
                assignments[i] = self.assign_centroid(data[i])

            old_centroids = [c.copy() for c in self.centroids]

            if self.clustering_algorithm == "kmean":
                self._adjust_centroid_kmean(data, assignments)
            elif self.clustering_algorithm == "kmedoid":
                self._adjust_centroid_kmedoid(data, assignments)
            else:
                raise ValueError(
                    f"Algoritmo desconocido: '{self.clustering_algorithm}'. "
                    "Usa 'kmean' o 'kmedoid'."
                )

            max_movement = max(
                self.euclidean_distance(old, new)
                for old, new in zip(old_centroids, self.centroids)
            )

            if max_movement < 1e-6:
                break

        return np.array(assignments), self.centroids

    def close(self):
        """Devuelve los centroides finales (codebook)."""
        return self.centroids


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.ml.quantizer import VectorQuantizer

    data = np.array([1, 2, 4, 5, 9, 10, 14, 18, 22], dtype=float).reshape(-1, 1)

    km = KClustering(n_centroids=3, clustering_algorithm="kmean")
    km.reset_centroids(dim=1)
    labels, centroids = km.clusterize(data)

    print("Datos originales:", data.ravel())
    print("Centroides finales:", np.array(centroids).ravel())
    print()

    for i in range(km.n_centroids):
        cluster_points = data[labels == i].ravel()
        print(f"Cluster {i}: {cluster_points}  (centroide={np.array(centroids[i]).ravel()})")

    q = VectorQuantizer(centroids)
    nuevo_punto = np.array([[7.0]])
    idx, vec = q.nearest_centroid(nuevo_punto)
    print(f"\nPunto nuevo {nuevo_punto.ravel()} -> cluster {idx} (centroide={vec})")
