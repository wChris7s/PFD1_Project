from data_reader.dataset_loader import DatasetLoader
from data_reader.dataset_types import DatasetType
from network_config.graph_topology import GraphTopology
from network_config.graph_types import GraphType


def main():
    graph = GraphTopology(GraphType.CYCLE_GRAPH, 20, 0.5)
    graph.create_graph()
    graph.show_graph()

    reader = DatasetLoader(DatasetType.MNIST)
    reader.load_data()

if __name__ == '__main__':
    main()