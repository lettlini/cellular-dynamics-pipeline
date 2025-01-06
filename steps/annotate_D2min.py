from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Optional

import networkx as nx
import numpy as np
from core_data_utils.datasets import BaseDataSet
from scipy.optimize import minimize
from tqdm import trange
from utils import get_future_label, get_object_positions


class D2minAnnotationTransformation:
    def __init__(
        self, mum_per_px: float, lag_time_frames: int, mininum_neighbors: int
    ) -> None:
        self._mum_per_px = mum_per_px
        self._lag_time_frames = lag_time_frames
        self._minimum_neighbors = mininum_neighbors

    @staticmethod
    def _calc_D2min(strain_tensor, dij_now, dij_future) -> float:

        # dij_now: array of shape (N, 2)
        # dij_future array of shape (N,2)

        assert dij_now.shape[0] == dij_future.shape[0]
        assert dij_now.shape[1] == 2
        assert dij_future.shape[1] == 2

        strain_tensor = np.reshape(strain_tensor, (2, 2))
        diff = dij_future - (dij_now @ strain_tensor.T)  # diff has shape (N, 2)

        assert diff.shape[0] == dij_now.shape[0]
        assert diff.shape[1] == 2
        return (diff**2).sum(axis=1).mean()

    def _annotate_single_node(
        self,
        graph_dataset: BaseDataSet,
        node_label: int,
        sindex: int,
        lag_time_frames: int,
    ):

        future_own_label = get_future_label(
            graph_dataset, node_label, lag_time_frames=lag_time_frames, sindex=sindex
        )

        if future_own_label is None:
            return np.NaN

        current_own_position = get_object_positions(
            graph_dataset, node_label, sindex=sindex
        )
        future_own_position = get_object_positions(
            graph_dataset, future_own_label, sindex=sindex + lag_time_frames
        )
        neighbor_current_positions = []
        neighbor_future_positions = []

        for nb in nx.neighbors(graph_dataset[sindex].data, node_label):
            future_nb_label = get_future_label(
                graph_dataset, nb, lag_time_frames, sindex
            )
            if future_nb_label is not None:
                neighbor_current_positions.append(
                    get_object_positions(graph_dataset, nb, sindex=sindex)
                )
                neighbor_future_positions.append(
                    get_object_positions(
                        graph_dataset, future_nb_label, sindex=sindex + lag_time_frames
                    )
                )

        if len(neighbor_current_positions) <= self._minimum_neighbors:
            return np.NaN

        neighbor_current_positions = np.vstack(neighbor_current_positions)
        neighbor_future_positions = np.vstack(neighbor_future_positions)

        dij_now = neighbor_current_positions - current_own_position
        dij_future = neighbor_future_positions - future_own_position

        min_result = minimize(
            D2minAnnotationTransformation._calc_D2min,
            x0=np.eye(2).reshape((4,)),
            args=(dij_now, dij_future),
            tol=1e-3,
        )
        if not min_result.success:
            raise RuntimeError("Minimize did not succeed.")

        D2min = min_result.fun.item() * (self._mum_per_px**2)

        return D2min

    def __call__(self, graph_dataset, property_suffix: Optional[str] = None) -> Any:

        psuff = "" if property_suffix is None else property_suffix
        graph_dataset = deepcopy(graph_dataset)

        for sindex in trange(len(graph_dataset)):
            for nodeid, nodeprops in graph_dataset[sindex].data.nodes(data=True):
                current_D2min = self._annotate_single_node(
                    graph_dataset=graph_dataset,
                    node_label=nodeid,
                    sindex=sindex,
                    lag_time_frames=self._lag_time_frames,
                )

                nodeprops[f"D2min_{psuff}"] = current_D2min

        return graph_dataset


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
        "--mum_per_px",
        required=True,
        type=float,
        help="Microns per pixel.",
    )
    parser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="Path to output file.",
    )
    parser.add_argument(
        "--minimum_neighbors",
        required=True,
        type=int,
        help="Minimum number of neighbors for D2min calculation.",
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
        x = D2minAnnotationTransformation(
            mum_per_px=args.mum_per_px,
            lag_time_frames=lt_frames,
            mininum_neighbors=args.minimum_neighbors,
        )(x, property_suffix=f"{lt_minutes}_minutes")

    x.to_pickle(args.outfile)
