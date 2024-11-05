import math

import numpy as np

from distributed_algorithms.utility import dist_subspace


class DistributedSangerAlgorithm:

    data:                   np.ndarray
    num_iterations:         int
    subspace_dimension:     int
    num_nodes:              int
    initial_estimate:       np.ndarray
    true_eigenspace:        np.ndarray

    def __init__(self, data: np.ndarray, iterations: int, subspace_dimension: int, num_nodes: int, initial_estimate: np.ndarray, true_eigenspace: np.ndarray):
        self.data = data
        self.num_iterations = iterations
        self.subspace_dimension = subspace_dimension
        self.num_nodes = num_nodes
        self.initial_estimate = initial_estimate
        self.true_eigenspace = true_eigenspace

    def execute(self, weight_matrix: np.ndarray, step_size: float, step_strategy: int) -> np.ndarray:
        # Error inicial
        error_angles = dist_subspace(self.true_eigenspace, self.initial_estimate)
        num_samples = self.data.shape[1]
        # Tamaño de la partición de datos por nodo
        partition_size = math.floor(num_samples / self.num_nodes)

        # Calcular matrices de covarianza locales para cada nodo
        local_covariance_matrices = self._calculate_local_covariance_matrices(partition_size)
        # Estimación actualizada
        current_estimate = np.tile(self.initial_estimate.transpose(), (self.num_nodes, 1))

        for iteration in range(self.num_iterations):
            # Ajustar el tamaño del paso de acuerdo a la estrategia seleccionada
            step_size_adjusted = self._adjust_step_size(step_size, step_strategy, iteration)
            # Actualizar la estimación usando la matriz de pesos y el gradiente de Sanger
            current_estimate = np.dot(weight_matrix, current_estimate) - step_size_adjusted * self._compute_sanger_gradient(local_covariance_matrices, current_estimate)
            # Actualizar la estimación usando la matriz de pesos y el gradiente de Sanger
            avg_error = self._calculate_average_error(current_estimate)
            error_angles = np.append(error_angles, avg_error)

        return error_angles

    def _calculate_local_covariance_matrices(self, partition_size: int) -> np.ndarray:
        # Calcula las matrices de covarianza locales para cada nodo basado en la partición de datos
        local_covariances = np.zeros((self.num_nodes,), dtype=object)
        for node_index in range(self.num_nodes):
            data_partition = self.data[:, node_index * partition_size:(node_index + 1) * partition_size]
            local_covariances[node_index] = (1 / partition_size) * np.dot(data_partition, data_partition.transpose())
        return local_covariances

    def _adjust_step_size(self, step_size: float, strategy: int, iteration: int) -> float:
        # Ajusta el tamaño del paso según la estrategia definida (constante, decreciente, etc.)
        if strategy == 0:
            # Estrategia 0: Utilizar un paso constante, mantiene el tamaño del paso igual durante todas las iteraciones
            return step_size  # Paso constante

        elif strategy == 1:
            # Estrategia 1: Utilizar un paso decreciente con una tasa moderada. Esto asegura una convergencia más estable.
            # El tamaño del paso decrece a medida que aumentan las iteraciones, reduciendo la magnitud del cambio en el subespacio estimado.
            return step_size / (iteration + 1) ** 0.2  # Paso decreciente

        elif strategy == 2:
            # Estrategia 2: Utilizar un paso decreciente más lento. Reduce el tamaño del paso usando la raíz cuadrada del número de iteraciones.
            # Este enfoque ofrece un decrecimiento más gradual del paso, asegurando una convergencia más lenta pero potencialmente más precisa.
            return step_size / math.sqrt(iteration + 1)  # Paso decreciente más lento

        return step_size

    def _compute_sanger_gradient(self, local_covariance_matrices: np.ndarray, current_estimate: np.ndarray) -> np.ndarray:
        # Calcula el gradiente de Sanger distribuido para actualizar la estimación del subespacio
        gradient = np.zeros(current_estimate.shape)  # Inicializar el gradiente con ceros
        for node_index in range(local_covariance_matrices.shape[0]):
            # Extraer la estimación local del nodo actual
            local_estimate = current_estimate[node_index * self.subspace_dimension:(node_index + 1) * self.subspace_dimension, :]

            # Transponer la estimación local
            transposed_estimate = local_estimate.transpose()

            # Calcular la matriz triangular superior usando el producto de la matriz de covarianza local y la estimación
            triangular_matrix = np.triu(np.dot(np.dot(local_estimate, local_covariance_matrices[node_index]), transposed_estimate))

            # Calcular la actualización del gradiente según la regla de Sanger
            # El término `-np.dot(local_covariance_matrices[node_index], transposed_estimate)` minimiza la proyección en la dirección incorrecta
            # El término `+ np.dot(transposed_estimate, triangular_matrix)` ajusta el gradiente para mantener la ortonormalidad
            gradient_update = -np.dot(local_covariance_matrices[node_index], transposed_estimate) + np.dot(transposed_estimate, triangular_matrix)

            # Almacenar la actualización del gradiente correspondiente para el nodo actual
            gradient[node_index * self.subspace_dimension:(node_index + 1) * self.subspace_dimension, :] = gradient_update.transpose()
        return gradient

    def _calculate_average_error(self, current_estimate: np.ndarray) -> float:
        # Calcula el error promedio entre la estimación actual y el subespacio verdadero
        error_sum = 0
        for node_index in range(self.num_nodes):
            local_estimate = current_estimate[node_index * self.subspace_dimension:(node_index + 1) * self.subspace_dimension, :]
            transposed_estimate = local_estimate.transpose()
            error_sum += dist_subspace(self.true_eigenspace, transposed_estimate)
        return error_sum / self.num_nodes