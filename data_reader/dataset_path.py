from enum import Enum

class DatasetPath(Enum):
    MNIST = "data_reader/dataset/raw/mnist_py3k.pkl.gz"
    CIFAR10 = "data_reader/dataset/raw/cifar-10-batches-py/data_batch"
    CIFAR10TEST = "data_reader/dataset/raw/cifar-10-batches-py/test_batch"