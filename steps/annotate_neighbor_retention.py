from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Optional

import networkx as nx
import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from tqdm import trange


class NeighborRetentionTransformation:

    def __init__(self, lag_time_frames: int) -> None:
        self._lag_time_frames = lag_time_frames

    @staticmethod
    def get_neighbor_list(G: nx.Graph, node: int) -> list[int]:
        return list(G.neighbors(node))

    @staticmethod
    def get_future_label(
        graph_ds: BaseDataSet, node_label: int, lag_time_frames: int, sindex: int
    ) -> int | None:

        if sindex + lag_time_frames + 1 >= len(graph_ds):
            raise ValueError(f"Not enough data.")
        if lag_time_frames == 0:
            return node_label

        current_label = node_label
        for i in range(lag_time_frames):
            nprops = graph_ds[sindex + i].data.nodes[current_label]
            if "next_object_id" not in nprops:
                return None
            current_label = nprops["next_object_id"]

        return current_label

    @staticmethod
    def get_neighbor_retention_fraction(
        graph_ds: BaseDataSet, node_label: int, lag_time_frames: int, sindex: int
    ) -> float:

        if sindex + lag_time_frames + 1 >= len(graph_ds):
            return np.NaN
        if lag_time_frames == 0:
            return 1.0

        future_own_label = NeighborRetentionTransformation.get_future_label(
            graph_ds=graph_ds,
            sindex=sindex,
            node_label=node_label,
            lag_time_frames=lag_time_frames,
        )

        if future_own_label is None:
            return np.NaN

        current_neighbors = NeighborRetentionTransformation.get_neighbor_list(
            graph_ds[sindex].data, node_label
        )
        if len(current_neighbors) == 0:
            return np.NaN

        # push neighbors forward
        future_current_neighbors = []

        for nb in current_neighbors:
            flabel = NeighborRetentionTransformation.get_future_label(
                graph_ds=graph_ds,
                sindex=sindex,
                node_label=nb,
                lag_time_frames=lag_time_frames,
            )
            if flabel is not None:
                future_current_neighbors.append(flabel)

        true_future_neighbors = NeighborRetentionTransformation.get_neighbor_list(
            graph_ds[sindex + lag_time_frames].data, future_own_label
        )
        retained_neighbors = 0

        for fnb in future_current_neighbors:
            if fnb in true_future_neighbors:
                retained_neighbors += 1

        return retained_neighbors / len(current_neighbors)

    def __call__(
        self, graph_ds: BaseDataSet, property_suffix: Optional[str] = None
    ) -> Any:

        graph_ds = deepcopy(graph_ds)
        suff = "" if property_suffix is None else property_suffix

        for i in trange(len(graph_ds)):
            for node, nodeprops in graph_ds[i].data.nodes(data=True):
                nodeprops[f"neighbor_retention_{suff}"] = (
                    NeighborRetentionTransformation.get_neighbor_retention_fraction(
                        graph_ds=graph_ds,
                        node_label=node,
                        lag_time_frames=self._lag_time_frames,
                        sindex=i,
                    )
                )

        return graph_ds


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--infile",
        required=True,
        type=str,
        help="Path to input file.",
    )
    parser.add_argument(
        "--lag_times_minutes",
        required=True,
        type=str,
        help="Comma separated list of lag times (in minutes)",
    )

    parser.add_argument(
        "--delta_t_minutes",
        required=True,
        type=int,
        help="Time difference between two adjacent frames.",
    )

    parser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="Path to output file.",
    )
    parser.add_argument(
        "--cpus",
        required=True,
        type=int,
        help="CPU cores to use.",
    )

    args = parser.parse_args()

    lag_times_minutes = [int(lt) for lt in args.lag_times_minutes.split(",")]
    lag_times_frames = [lt // args.delta_t_minutes for lt in lag_times_minutes]

    x = BaseDataSet.from_pickle(args.infile)

    for lt_frames, lt_minutes in zip(lag_times_frames, lag_times_minutes, strict=True):
        x = NeighborRetentionTransformation(lt_frames)(
            x, property_suffix=f"{lt_minutes}_minutes"
        )

    x.to_pickle(args.outfile)
