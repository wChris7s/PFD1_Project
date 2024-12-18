import math

import numpy as np

from distributed_algorithms.utility import adjust_step_size
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
        error_angles = dist_subspace(self.true_eigenspace, self.initial_estimate)
        num_samples = self.data.shape[1]
        partition_size = math.floor(num_samples / self.num_nodes)
        local_covariance_matrices = self._calculate_local_covariance_matrices(partition_size)
        current_estimate = np.tile(self.initial_estimate.transpose(), (self.num_nodes, 1))

        for iteration in range(self.num_iterations):
            step_size_adjusted = adjust_step_size(step_size, step_strategy, iteration)
            current_estimate = np.dot(weight_matrix, current_estimate) - step_size_adjusted * self._compute_sanger_gradient(local_covariance_matrices, current_estimate)
            avg_error = self._calculate_average_error(current_estimate)
            error_angles = np.append(error_angles, avg_error)

        return error_angles

    def _calculate_local_covariance_matrices(self, partition_size: int) -> np.ndarray:
        local_covariances = np.zeros((self.num_nodes,), dtype=object)
        for node_index in range(self.num_nodes):
            data_partition = self.data[:, node_index * partition_size:(node_index + 1) * partition_size]
            local_covariances[node_index] = (1 / partition_size) * np.dot(data_partition, data_partition.transpose())
        return local_covariances

    def _compute_sanger_gradient(self, local_covariance_matrices: np.ndarray, current_estimate: np.ndarray) -> np.ndarray:
        gradient = np.zeros(current_estimate.shape)
        for node_index in range(local_covariance_matrices.shape[0]):
            local_estimate = current_estimate[node_index * self.subspace_dimension:(node_index + 1) * self.subspace_dimension, :]
            transposed_estimate = local_estimate.transpose()
            triangular_matrix = np.triu(np.dot(np.dot(local_estimate, local_covariance_matrices[node_index]), transposed_estimate))
            gradient_update = -np.dot(local_covariance_matrices[node_index], transposed_estimate) + np.dot(transposed_estimate, triangular_matrix)
            gradient[node_index * self.subspace_dimension:(node_index + 1) * self.subspace_dimension, :] = gradient_update.transpose()
        return gradient

    def _calculate_average_error(self, current_estimate: np.ndarray) -> float:
        error_sum = 0
        for node_index in range(self.num_nodes):
            local_estimate = current_estimate[node_index * self.subspace_dimension:(node_index + 1) * self.subspace_dimension, :]
            transposed_estimate = local_estimate.transpose()
            error_sum += dist_subspace(self.true_eigenspace, transposed_estimate)
        return error_sum / self.num_nodes