from data_reader.dataset_loader import DatasetLoader
from data_reader.dataset_types import DatasetType
from distributed_algorithms.dsa_algorithm import \
    DistributedSangerAlgorithm as dsa
from distributed_algorithms.utility import initialize_parameters
from network_config.graph_topology import GraphTopology
from network_config.graph_types import GraphType


def main():

    k = 5
    iterations = 10000
    alpha = 0.1
    number_of_nodes = 10
    prob_connectivity = 0.5
    step_strategy = 0

    graph = GraphTopology(GraphType.CYCLE_GRAPH, number_of_nodes,
                          prob_connectivity)
    graph.create_graph()

    reader = DatasetLoader(DatasetType.MNIST)
    reader.load_data()

    weight_matrix, initial_estimate, true_eigenspace = initialize_parameters(graph, reader, k)

    dsa_algorithm = dsa(data=reader.get_data(),
                        iterations=iterations,
                        subspace_dimension=k,
                        num_nodes=number_of_nodes,
                        initial_estimate=initial_estimate,
                        true_eigenspace=true_eigenspace)

    angle_dsa = dsa_algorithm.execute(weight_matrix=weight_matrix,
                                      step_size=alpha,
                                      step_strategy=step_strategy)
    print(angle_dsa)

if __name__ == '__main__':
    main()
