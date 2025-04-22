import networkx as nx
import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.transformations import BaseDataSetTransformation
from argparse import ArgumentParser


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


class AnnotateKFoldNeighborhoodDensity(BaseDataSetTransformation):
    def __init__(self, k):
        self._k = k
        super().__init__()

    def _transform_single_entry(self, entry, dataset_properties):
        # here we want to annotate each node with its k-order neighborhood density

        graph: nx.Graph = entry.data.copy()

        for node in graph.nodes:
            areas = []

            for ck in range(self._k + 1):
                for cnb in get_k_fold_neighbors(graph, node, ck):
                    areas.append(graph.nodes[cnb]["cell_area_mum_squared"])
                    graph.nodes[node][f"{ck}-order_local_density_per_mum_squared"] = (
                        1 / np.mean(np.array(areas))
                    )

        return BaseDataSetEntry(identifier=entry.identifier, data=graph)


if __name__ == "__main__":

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
    x = AnnotateKFoldNeighborhoodDensity(k=7)(x, cpus=args.cpus)

    x.to_pickle(args.outfile)
