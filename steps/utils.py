import networkx as nx
import numpy as np
from core_data_utils.datasets import BaseDataSet


def get_neighbor_list(G: nx.Graph, node: int) -> list[int]:
    return list(G.neighbors(node))


def get_object_positions(graph_ds, node_label, sindex, prefix="cell"):
    x = graph_ds[sindex].data.nodes[node_label][f"{prefix}_centroid_x"]
    y = graph_ds[sindex].data.nodes[node_label][f"{prefix}_centroid_y"]

    return np.array([x, y])


def get_future_label(
    graph_ds: BaseDataSet, node_label: int, lag_time_frames: int, sindex: int
) -> int | None:

    if sindex + lag_time_frames + 1 >= len(graph_ds):
        return None
    if lag_time_frames == 0:
        return node_label

    current_label = node_label
    for i in range(lag_time_frames):
        nprops = graph_ds[sindex + i].data.nodes[current_label]
        if "next_object_id" not in nprops:
            return None
        current_label = nprops["next_object_id"]

    return current_label
