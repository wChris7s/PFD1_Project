import numpy as np
import pickle

from data_reader.dataset_loader import DatasetLoader
from network_config.graph_topology import GraphTopology


def dist_subspace(x, y):
    x = x / np.linalg.norm(x, axis=0)
    y = y / np.linalg.norm(y, axis=0)
    m = np.matmul(x.transpose(), y)
    sine_angle = 1 - np.diag(m) ** 2
    dist = np.sum(sine_angle) / x.shape[1]
    return dist

def initialize_parameters(graph: GraphTopology, reader: DatasetLoader, subspace_dimension: int):
    # Obtiene la matriz de pesos del grafo y formar la matriz de pesos para la red
    weight_matrix = np.kron(graph.get_weight_matrix(), np.identity(subspace_dimension))

    # Carga los eigenvectores verdaderos
    with open(f"data_reader/dataset/true_eigenvectors/EV_{reader.get_dataset_name().value}.pickle", 'rb') as f:
        true_eigenvectors = pickle.load(f)
    true_eigenspace = true_eigenvectors[:, 0:subspace_dimension]

    # Inicializa aleatoriamente los valores iniciales del subespacio
    np.random.seed(1)
    initial_estimate = np.random.rand(reader.get_data().shape[0], subspace_dimension)
    # Cálculo de la descomposición QR (initial_estimate => ortonormal)
    initial_estimate, _ = np.linalg.qr(initial_estimate)

    return weight_matrix, initial_estimate, true_eigenspace