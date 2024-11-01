from enum import Enum

class GraphType(Enum):
    ERDOS_RENYI_GRAPH = "erdos_renyi_graph"
    CYCLE_GRAPH = "cycle"
    MARGULIS_GABBER_GALIL_GRAPH = "expander"
    STAR_GRAPH = "star"