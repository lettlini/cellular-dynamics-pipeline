from torch_geometric.data import Data
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.transformations import BaseDataSetTransformation, BaseFilter
from torch_geometric.utils import from_networkx
import networkx as nx
from typing import Iterable
from argparse import ArgumentParser
import multiprocessing as mp
import torch


class Nx2TorchTransformation(BaseDataSetTransformation):
    def __init__(
        self,
        target_properties: Iterable[str],
        node_properties: Iterable[str],
        edge_properties: Iterable[str],
    ):
        self._target_props = target_properties
        self._node_props = node_properties
        self._edge_props = edge_properties
        super().__init__()

    def _transform_single_entry(self, entry, _):
        nx_graph = entry.data

        for _, ndat in nx_graph.nodes(data=True):
            for k in list(ndat.keys()):
                if (k not in self._node_props) and (k not in self._target_props):
                    del ndat[k]

        ndat = from_networkx(
            nx_graph,
            group_node_attrs=self._node_props,
            group_edge_attrs=self._edge_props,
        )
        ndat.y = torch.hstack(
            [torch.reshape(ndat[tprop], shape=(-1, 1)) for tprop in self._target_props]
        )

        for tprop in self._target_props:
            del ndat[tprop]

        return BaseDataSetEntry(
            identifier=entry.identifier, data=ndat, metadata=entry.metadata
        )


if __name__ == "__main__":

    mp.set_start_method("spawn")

    parser = ArgumentParser()
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--torch_node_properties", type=str, required=True)
    parser.add_argument("--torch_edge_properties", type=str, required=True)
    parser.add_argument("--torch_target_properties", type=str, required=True)

    parser.add_argument(
        "--cpus",
        required=True,
        type=int,
        help="CPU cores to use.",
    )

    args = parser.parse_args()

    node_props: list[str] = args.torch_node_properties.split(",")
    edge_props: list[str] = args.torch_edge_properties.split(",")
    target_props: list[str] = args.torch_target_properties.split(",")

    x = BaseDataSet.from_pickle(args.infile)

    x = Nx2TorchTransformation(
        node_properties=node_props,
        edge_properties=edge_props,
        target_properties=target_props,
    )(x)

    x.to_pickle(args.outfile)
