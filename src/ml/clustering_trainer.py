import numpy as np


class KClustering:
    """Agrupa vectores en clusters usando k-means o k-medoids (numpy vectorizado).

    Uso:
        kc = KClustering(n_centroids=5, clustering_algorithm="kmean")
        kc.reset_centroids(dim=13)
        labels, centroids = kc.clusterize(feature_matrix)
    """

    def __init__(self, n_centroids=100, clustering_algorithm="kmean",
                 batch_size: int = 100_000, max_iter: int = 100,
                 tol: float = 1e-6):
        self.centroids = np.empty((0, 0), dtype=np.float32)
        self.n_centroids = n_centroids
        self.clustering_algorithm = clustering_algorithm
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol

    def reset_centroids(self, dim=128):
        """Inicializa centroides con valores aleatorios."""
        self.centroids = np.random.randn(self.n_centroids, dim).astype(np.float32)

    def euclidean_distance(self, vector_a, vector_b):
        """Distancia euclidiana entre dos vectores (compat con API previa)."""
        a = np.asarray(vector_a, dtype=np.float32)
        b = np.asarray(vector_b, dtype=np.float32)
        diff = a - b
        return float(np.sqrt(np.sum(diff * diff)))

    def _assign_all(self, data: np.ndarray) -> np.ndarray:
        """Asigna cada fila al centroide más cercano, en bloques."""
        C = self.centroids
        c_sq = np.sum(C ** 2, axis=1)
        out = np.empty(len(data), dtype=np.int64)
        for start in range(0, len(data), self.batch_size):
            end = min(start + self.batch_size, len(data))
            Xb = data[start:end]
            x_sq = np.sum(Xb ** 2, axis=1, keepdims=True)
            dists_sq = x_sq + c_sq[None, :] - 2.0 * (Xb @ C.T)
            out[start:end] = dists_sq.argmin(axis=1)
        return out

    def assign_centroid(self, vector):
        """Indice del centroide más cercano (compat con API previa)."""
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        return int(self._assign_all(v)[0])

    def _adjust_centroid_kmean(self, data, assignments):
        """Actualiza centroides como promedio de sus puntos (vectorizado)."""
        D = data.shape[1]
        new_centroids = np.zeros((self.n_centroids, D), dtype=np.float32)
        counts = np.bincount(assignments, minlength=self.n_centroids)
        np.add.at(new_centroids, assignments, data)
        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty, None]
        # clusters vacios: conservar centroide anterior
        new_centroids[~nonempty] = self.centroids[~nonempty]
        self.centroids = new_centroids

    def _adjust_centroid_kmedoid(self, data, assignments):
        """Actualiza centroides como el punto más central del cluster.

        Para cada cluster vectorizamos las distancias pairwise entre sus
        miembros y elegimos el de menor suma. Coste O(B^2 * D) por cluster,
        donde B = tamaño del cluster.
        """
        new_centroids = self.centroids.copy()
        for k in range(self.n_centroids):
            mask = assignments == k
            if not mask.any():
                continue
            members = data[mask]
            # dist^2 = ||a||^2 + ||b||^2 - 2 a.b
            sq = np.sum(members ** 2, axis=1)
            cross = members @ members.T
            dists_sq = sq[:, None] + sq[None, :] - 2.0 * cross
            np.maximum(dists_sq, 0, out=dists_sq)
            dists = np.sqrt(dists_sq)
            costs = dists.sum(axis=1)
            new_centroids[k] = members[int(costs.argmin())]
        self.centroids = new_centroids

    def clusterize(self, matrix):
        """Ejecuta el algoritmo de clustering hasta convergencia."""
        data = np.asarray(matrix, dtype=np.float32)
        n_points = len(data)
        self.n_centroids = min(self.n_centroids, n_points)
        if isinstance(self.centroids, list):
            self.centroids = np.asarray(self.centroids, dtype=np.float32)
        if len(self.centroids) > self.n_centroids:
            self.centroids = self.centroids[: self.n_centroids]

        assignments = np.zeros(n_points, dtype=np.int64)
        for _ in range(self.max_iter):
            assignments = self._assign_all(data)
            old = self.centroids.copy()

            if self.clustering_algorithm == "kmean":
                self._adjust_centroid_kmean(data, assignments)
            elif self.clustering_algorithm == "kmedoid":
                self._adjust_centroid_kmedoid(data, assignments)
            else:
                raise ValueError(
                    f"Algoritmo desconocido: '{self.clustering_algorithm}'. "
                    "Usa 'kmean' o 'kmedoid'."
                )

            movement = float(np.max(np.linalg.norm(self.centroids - old, axis=1)))
            if movement < self.tol:
                break

        return assignments, self.centroids

    def close(self):
        """Devuelve los centroides finales (lista de 1D para compat)."""
        return [self.centroids[i] for i in range(len(self.centroids))]


if __name__ == "__main__":
    import os
    import sys

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
