import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from network_config.graph_types import GraphType


class GraphTopology:
    __graph_type: GraphType
    __number_of_nodes: int
    __prob_connectivity: float
    __weight_matrix: np.ndarray
    __graph: nx.Graph
    __is_connected: bool

    def __init__(self, graph_type: GraphType, number_of_nodes: int, prob_connectivity: float):
        self.__graph_type = graph_type
        self.__number_of_nodes = number_of_nodes
        self.__prob_connectivity = prob_connectivity
        self.__is_connected = False

    def create_graph(self) -> None:
        self.__graph = self.__generate_graph()
        self.__is_connected = nx.is_connected(self.__graph)
        self.__create_metropolis_weight_matrix()

    def __generate_graph(self) -> nx.Graph:
        if self.__graph_type == GraphType.ERDOS_RENYI_GRAPH:
            return self.__generate_erdos_renyi_graph()
        elif self.__graph_type == GraphType.CYCLE_GRAPH:
            return self.__generate_cycle_graph()
        elif self.__graph_type == GraphType.MARGULIS_GABBER_GALIL_GRAPH:
            return self.__generate_margulis_gabber_galil_graph()
        elif self.__graph_type == GraphType.STAR_GRAPH:
            return self.__generate_star_graph()
        else:
            raise TypeError("The type of graph is not correct.")

    def __generate_erdos_renyi_graph(self) -> nx.Graph:
        while True:
            graph = nx.erdos_renyi_graph(self.__number_of_nodes, self.__prob_connectivity)
            if nx.is_connected(graph):
                return graph

    def __generate_cycle_graph(self) -> nx.Graph:
        while True:
            graph = nx.cycle_graph(self.__number_of_nodes)
            if nx.is_connected(graph):
                return graph

    def __generate_margulis_gabber_galil_graph(self) -> nx.Graph:
        while True:
            graph = nx.margulis_gabber_galil_graph(self.__number_of_nodes)
            if nx.is_connected(graph):
                return graph

    def __generate_star_graph(self) -> nx.Graph:
        while True:
            graph = nx.star_graph(self.__number_of_nodes - 1)
            if nx.is_connected(graph):
                return graph

    def __create_metropolis_weight_matrix(self) -> None:
        adjacency_matrix = nx.to_numpy_array(self.__graph)
        node_degrees = self.__calculate_node_degrees(adjacency_matrix)
        self.__weight_matrix = self.__calculate_metropolis_weights(adjacency_matrix, node_degrees)

    @staticmethod
    def __calculate_node_degrees(adjacency_matrix: np.ndarray) -> np.ndarray:
        degrees = np.sum(adjacency_matrix, axis=1)
        return degrees.astype(np.int64)

    @staticmethod
    def __calculate_metropolis_weights(adjacency_matrix: np.ndarray, node_degrees: np.ndarray) -> np.ndarray:
        weight_matrix = np.zeros(adjacency_matrix.shape)
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i, adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] != 0 and i != j:
                    weight_matrix[i, j] = 1 / (max(node_degrees[i], node_degrees[j]) + 1)
                    weight_matrix[j, i] = weight_matrix[i, j]
        for i in range(adjacency_matrix.shape[0]):
            weight_matrix[i, i] = 1 - np.sum(weight_matrix[i, :])
        return weight_matrix

    def show_graph(self) -> None:
        if self.__graph is None:
            raise ValueError("Graph has not been created yet. Please call create_graph() first.")
        nx.draw(self.__graph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.show()

    def get_weight_matrix(self) -> np.ndarray:
        if self.__graph is None:
            raise ValueError("Graph has not been created yet. Please call create_graph() first.")
        return self.__weight_matrix

    def is_connected(self) -> bool:
        return self.__is_connected