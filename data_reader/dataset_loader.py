import gzip
import math
import pickle
from typing import Optional

import numpy as np

from data_reader.dataset_path import DatasetPath
from data_reader.dataset_types import DatasetType


class DatasetLoader:
    __dataset_name: DatasetType
    __data: Optional[np.ndarray]

    def __init__(self, dataset_name: DatasetType):
        self.__dataset_name = dataset_name
        self.__data: Optional[np.ndarray] = None

    def load_data(self) -> None:
        if self.__dataset_name == DatasetType.MNIST:
            self.__data = self.__load_mnist_data()
        elif self.__dataset_name == DatasetType.CIFAR10:
            self.__data = self.__load_cifar10_data()
        else:
            raise ValueError("Dataset name not recognized. Please use 'mnist' or 'cifar10'.")

    def __load_mnist_data(self) -> np.ndarray:
        mnist_data = self.__read_mnist_file(DatasetPath.MNIST.value)
        combined_data = self.__combine_mnist_data(mnist_data)
        centered_data = self.__center_data(combined_data)
        return centered_data

    def __load_cifar10_data(self) -> np.ndarray:
        training_data = self.__load_cifar10_batches()
        test_data = self.__load_cifar10_test_batch()
        all_data = self.__concatenate_data(training_data, test_data)
        selected_features_data = self.__select_cifar10_features(all_data)
        centered_data = self.__center_data(selected_features_data)
        normalized_data = self.__normalize_data(centered_data)
        return normalized_data

    @staticmethod
    def __read_mnist_file(file_path: str) -> tuple:
        with gzip.open(file_path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def __combine_mnist_data(mnist_data: tuple) -> np.ndarray:
        (train_inputs, _), (valid_inputs, _), (_, _) = mnist_data
        combined_data = np.concatenate((train_inputs, valid_inputs))
        return combined_data.transpose()

    def __load_cifar10_batches(self) -> np.ndarray:
        data_concatenated = []

        for i in range(1, 6):
            file_path = f'{DatasetPath.CIFAR10.value}_{i}'
            batch_data = self.__unpickle(file_path)[b'data']
            data_concatenated = self.__concatenate_data(data_concatenated, batch_data)

        return data_concatenated

    def __load_cifar10_test_batch(self) -> np.ndarray:
        test_file_path = DatasetPath.CIFAR10TEST.value
        return self.__unpickle(test_file_path)[b'data']

    @staticmethod
    def __concatenate_data(training_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
        if len(training_data) == 0:
            return test_data
        else:
            return np.concatenate((training_data, test_data))

    @staticmethod
    def __select_cifar10_features(data: np.ndarray) -> np.ndarray:
        data = data[:, :1024]
        return data.transpose()

    @staticmethod
    def __normalize_data(data: np.ndarray) -> np.ndarray:
        num_samples = data.shape[1]
        covariance_matrix = (1 / num_samples) * np.dot(data, data.transpose())
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        largest_eigenvalue = np.flip(eigenvalues)[0]
        return data / math.sqrt(largest_eigenvalue)

    @staticmethod
    def __unpickle(file: str) -> dict:
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')

    @staticmethod
    def __center_data(data: np.ndarray) -> np.ndarray:
        dimension = data.shape[0]
        num_samples = data.shape[1]
        mean_vector = np.mean(data, axis=1).reshape(dimension, 1)
        mean_matrix = np.tile(mean_vector, (1, num_samples))
        return data - mean_matrix

    def has_data(self) -> bool:
        return self.__data is not None

    def get_data(self) -> np.ndarray:
        if self.__data is None:
            raise ValueError("Data has not been loaded yet. Please call load_data() first.")
        return self.__data
