import numpy as np


class VectorQuantizer:
    """Cuenta cuántos vectores caen en cada centroide del codebook."""

    def __init__(self, centroids):
        """Guarda los centroides del codebook."""
        self.centroids = [np.asarray(c) for c in centroids]
        self.n_centroids = len(centroids)

    def nearest_centroid(self, vector):
        """Devuelve (índice, centroide) más cercano al vector."""
        best_dist = float('inf')
        best_idx = 0
        for i, centroid in enumerate(self.centroids):
            diff = np.asarray(vector) - centroid
            dist = float(np.sqrt(np.sum(diff * diff)))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx, self.centroids[best_idx]

    def histogram(self, matrix):
        """Asigna cada vector al centroide más cercano y cuenta frecuencias."""
        data = np.asarray(matrix)
        counts = np.zeros(self.n_centroids, dtype=int)
        for row in data:
            idx, _ = self.nearest_centroid(row)
            counts[idx] += 1
        return counts


class WordQuantizer:
    """Cuenta cuántas veces aparece cada palabra del codebook en un texto."""

    def __init__(self, bag_of_words):
        """Guarda el vocabulario de referencia."""
        self.bag_of_words = list(bag_of_words)
        self.n_words = len(bag_of_words)

    def histogram(self, tokens):
        """Cuenta ocurrencias de cada palabra del codebook en los tokens."""
        counts = np.zeros(self.n_words, dtype=int)
        for token in tokens:
            for i, word in enumerate(self.bag_of_words):
                if token == word:
                    counts[i] += 1
                    break
        return counts


if __name__ == "__main__":
    centroids = np.array([[0.0], [5.0], [10.0]])
    vq = VectorQuantizer(centroids)

    vec = np.array([4.0])
    idx, c = vq.nearest_centroid(vec)
    print(f"Vector {vec} -> cluster {idx}, centroide={c}")

    matrix = np.array([[1.0], [2.0], [6.0], [9.0], [11.0]])
    hist = vq.histogram(matrix)
    print(f"Histograma vector: {hist}")

    bow = ["hola", "mundo", "test", "foo"]
    wq = WordQuantizer(bow)
    tokens = ["hola", "hola", "foo", "bar", "test"]
    wh = wq.histogram(tokens)
    print(f"Bag of words: {bow}")
    print(f"Tokens: {tokens}")
    print(f"Histograma palabras: {wh}")
