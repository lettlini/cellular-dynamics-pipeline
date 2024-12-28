import inspect
from argparse import ArgumentParser

import networkx as nx
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.transformations import BaseDataSetTransformation


class GraphTheoreticalAnnotationsTransform(BaseDataSetTransformation):

    def _transform_single_entry(self, entry: BaseDataSetEntry) -> BaseDataSetEntry:
        cgraph = entry.data

        # Find all methods that start with 'attribute_node'
        methods = [
            (name, method)
            for name, method in inspect.getmembers(self, predicate=inspect.ismethod)
            if name.startswith("_node_property_") or name.startswith("_edge_property")
        ]

        for name, method in methods:
            if name.startswith("_edge_property_"):
                current_property_name = name.removeprefix("_edge_property_")

                assert not any(
                    current_property_name in data
                    for _, __, data in cgraph.edges(data=True)
                )

                nx.set_edge_attributes(cgraph, method(cgraph), current_property_name)

            elif name.startswith("_node_property_"):
                current_property_name = name.removeprefix("_node_property_")

                assert not any(
                    current_property_name in data for _, data in cgraph.nodes(data=True)
                )

                nx.set_node_attributes(cgraph, method(cgraph), current_property_name)

        return BaseDataSetEntry(identifier=entry.identifier, data=cgraph)

    def _node_property_betweenness_centrality(self, graph: nx.Graph) -> dict:
        return nx.betweenness_centrality(graph)

    def _node_property_closeness_centrality(self, graph: nx.Graph) -> dict:
        return nx.closeness_centrality(graph)

    def _node_property_local_clustering_coefficient(self, graph: nx.Graph) -> dict:
        return nx.clustering(graph)

    def _node_property_degree(self, graph: nx.Graph) -> dict:
        return dict(nx.degree(graph))

    def _edge_property_betweenness_centrality(self, graph: nx.Graph) -> dict:
        return nx.edge_betweenness_centrality(graph)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--infile",
        required=True,
        type=str,
        help="Path to input file.",
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
        help="Path to output file.",
    )

    args = parser.parse_args()

    x = BaseDataSet.from_pickle(args.infile)

    x = GraphTheoreticalAnnotationsTransform()(dataset=x, parallel=True, cpus=args.cpus)

    x.to_pickle(args.outfile)
