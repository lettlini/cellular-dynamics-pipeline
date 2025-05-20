import networkx as nx
import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.transformations import BaseDataSetTransformation
from argparse import ArgumentParser
import multiprocessing as mp


def get_k_fold_neighbors(G, node, k):
    """
    Returns the set of k-fold neighbors of a node.

    Parameters:
    G : NetworkX graph
    node : node in the graph
    k : int
        The exact distance to consider

    Returns:
    set: Nodes exactly k steps away from the given node
    """
    if k < 0:
        raise ValueError("k must be a non-negative integer")

    if k == 0:
        return {node}

    # Get all nodes within distance k
    nodes_within_k = nx.single_source_shortest_path_length(G, node, cutoff=k)

    # Filter for only those at exactly distance k
    k_fold_neighbors = {n for n, dist in nodes_within_k.items() if dist == k}

    return k_fold_neighbors


class AnnotateKFoldNeighborhoodAverage(BaseDataSetTransformation):
    def __init__(self, k, prefix: str, attribute: str):
        self._k = k
        self._prefix = prefix
        self._attribute = attribute

        super().__init__()

    def _k_scaling(self, k: int) -> float:
        raise NotImplementedError("k scaling function has not yet been implemented")

    def _transform_single_entry(self, entry, dataset_properties):
        # here we want to annotate each node with its k-order neighborhood average

        graph: nx.Graph = entry.data

        for node in graph.nodes:
            avals = []
            weights = []

            for ck in range(self._k + 1):
                k_scaler: float = self._k_scaling(ck)

                for cnb in get_k_fold_neighbors(graph, node, ck):
                    avals.append(graph.nodes[cnb][self._attribute])
                    weights.append(k_scaler)

                graph.nodes[node][f"{self._prefix}-{ck}-order_{self._attribute}"] = (
                    np.average(np.array(avals), weights=np.array(weights))
                )

        return BaseDataSetEntry(identifier=entry.identifier, data=graph)


class StaticKFoldNeighborhoodDensity(AnnotateKFoldNeighborhoodAverage):
    def __init__(self, k, attribute):
        super().__init__(k, "static", attribute=attribute)

    def _k_scaling(self, k):
        return 1.0


class InverseLinearKFoldNeighborhoodDensity(AnnotateKFoldNeighborhoodAverage):
    def __init__(self, k, attribute):
        super().__init__(k, "inverse_linear", attribute=attribute)

    def _k_scaling(self, k):
        return 1.0 / max(1, k)


class InverseQuadraticKFoldNeighborhoodDensity(AnnotateKFoldNeighborhoodAverage):
    def __init__(self, k, attribute):
        super().__init__(k, "inverse_quadratic", attribute=attribute)

    def _k_scaling(self, k):
        return 1.0 / (max(k, 1) ** 2)


if __name__ == "__main__":

    mp.set_start_method("spawn")

    parser = ArgumentParser()
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument(
        "--cpus",
        required=True,
        type=int,
        help="CPU cores to use.",
    )

    args = parser.parse_args()

    x = BaseDataSet.from_pickle(args.infile)

    x = StaticKFoldNeighborhoodDensity(k=7, attribute="cell_area_mum_squared")(
        x, cpus=args.cpus
    )
    x = InverseLinearKFoldNeighborhoodDensity(k=7, attribute="cell_area_mum_squared")(
        x, cpus=args.cpus
    )
    x = InverseQuadraticKFoldNeighborhoodDensity(
        k=7, attribute="cell_area_mum_squared"
    )(x, cpus=args.cpus)

    for attr in ["cell_shape", "nucleus_shape"]:
        x = StaticKFoldNeighborhoodDensity(k=7, attribute=attr)(x, cpus=args.cpus)
        x = InverseLinearKFoldNeighborhoodDensity(k=7, attribute=attr)(
            x, cpus=args.cpus
        )
        x = InverseQuadraticKFoldNeighborhoodDensity(k=7, attribute=attr)(
            x, cpus=args.cpus
        )

    x.to_pickle(args.outfile)
