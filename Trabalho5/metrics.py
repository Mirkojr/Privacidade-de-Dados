import numpy as np

def distancia_euclidiana(data_set: np.ndarray, x_ponto: np.ndarray) -> np.ndarray:
    """Calcula a distância euclidiana vetorizada."""
    return np.linalg.norm(data_set - x_ponto, axis=1)