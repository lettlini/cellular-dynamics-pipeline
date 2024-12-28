from argparse import ArgumentParser

import cv2
import networkx as nx
import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.transformations import BaseDataSetTransformation


class BuildGraphTransform(BaseDataSetTransformation):

    def __init__(
        self,
        mum_px: float,
    ) -> None:
        self._mum_per_px = mum_px
        super().__init__()

    def _transform_single_entry(self, entry: BaseDataSetEntry) -> BaseDataSetEntry:

        properties = entry.data

        G = nx.Graph()

        # add all nodes in a first pass
        for cell_id, property_dict in properties.items():
            if not cell_id in G:
                G.add_node(
                    cell_id,
                    **{k: v for k, v in property_dict.items() if k != "neighbors"},
                )
            else:
                raise RuntimeError(f"Node {cell_id} already present in graph")

        # add all edges in a second pass
        for cell_id, property_dict in properties.items():
            for neighbor_index, overlap in property_dict["neighbors"].items():
                oc_x, oc_y = (
                    property_dict["cell_centroid_x"],
                    property_dict["cell_centroid_y"],
                )
                nbc_x, nbc_y = (
                    properties[neighbor_index]["cell_centroid_x"],
                    properties[neighbor_index]["cell_centroid_y"],
                )

                own_centroid = np.array([oc_x, oc_y])
                nbh_centroid = np.array([nbc_x, nbc_y])

                distance = (
                    np.linalg.norm(own_centroid - nbh_centroid) * self._mum_per_px
                )

                # edge has been created before
                if G.has_edge(cell_id, neighbor_index):
                    old_value = G[cell_id][neighbor_index]["shared_cell_perimeter_mum"]
                    new_value = (old_value + overlap) / 2
                    G[cell_id][neighbor_index]["shared_cell_perimeter_mum"] = new_value

                    # check if distance is consistent:
                    if not np.allclose(
                        distance, G[cell_id][neighbor_index]["distance_mum"]
                    ):
                        raise RuntimeError(
                            f"New distance {distance} inconsistent with old distance {G[cell_id][neighbor_index]['distance_mum']}"
                        )
                else:
                    G.add_edge(
                        cell_id,
                        neighbor_index,
                        shared_cell_perimeter_mum=overlap,
                        distance_mum=distance,
                    )

        return BaseDataSetEntry(identifier=entry.identifier, data=G)


class CalculateCellNucleusShapeTransformation(BaseDataSetTransformation):

    def _transform_single_entry(self, entry: BaseDataSetEntry) -> BaseDataSetEntry:
        graph = entry.data

        for _, nodeprops in graph.nodes(data=True):
            nodeprops["cell_shape"] = nodeprops["cell_perimeter_mum"] / np.sqrt(
                nodeprops["cell_area_mum_squared"]
            )
            nodeprops["nucleus_shape"] = (
                nodeprops["nucleus_major_axis_mum"]
                / nodeprops["nucleus_minor_axis_mum"]
            )

            assert nodeprops["nucleus_shape"] >= 1.0
            assert nodeprops["cell_shape"] >= 2 / np.sqrt(np.pi)

        return BaseDataSetEntry(identifier=entry.identifier, data=graph)


class CalculateOrderParameter(BaseDataSetTransformation):

    def __init__(self, angle_prop: str, save_prop_prefix: str) -> None:
        self._angle_property = angle_prop
        self._save_property_prefix = save_prop_prefix

        super().__init__()

    @staticmethod
    def calculate_Q_tensor_entry(data, indices: tuple[int, int]) -> float:
        i, j = indices
        delta = 1.0 if (i == j) else 0.0

        return 2 * (np.multiply(data[:, i], data[:, j]) - 0.5 * delta).mean()

    @staticmethod
    def calculate_order_parameter(data: list[np.array]) -> float:

        Q_tensor = np.zeros((2, 2))

        data_matrix = np.vstack(data)

        # assert that vectors have unit norm
        assert np.allclose(np.linalg.norm(data_matrix, axis=1), 1.0)

        Q_tensor[0, 0] = CalculateOrderParameter.calculate_Q_tensor_entry(
            data_matrix, (0, 0)
        )
        Q_tensor[1, 0] = CalculateOrderParameter.calculate_Q_tensor_entry(
            data_matrix, (1, 0)
        )
        Q_tensor[0, 1] = Q_tensor[1, 0]  # tensor is symmetric!
        Q_tensor[1, 1] = CalculateOrderParameter.calculate_Q_tensor_entry(
            data_matrix, (1, 1)
        )

        S = np.max(np.linalg.eigvals(Q_tensor))

        return S, Q_tensor

    def _transform_single_entry(self, entry: BaseDataSetEntry) -> BaseDataSetEntry:

        current_graph = entry.data

        # calculate the nematic order parameter in a second pass
        for nodeid, nodedata in current_graph.nodes(data=True):

            # collect all local directors
            own_angle = nodedata[self._angle_property]

            neighborhood_local_directors = [
                np.array([np.cos(own_angle), np.sin(own_angle)])
            ]

            for neighbor in current_graph.neighbors(nodeid):
                cangle = current_graph.nodes[neighbor][self._angle_property]
                neighborhood_local_directors.append(
                    np.array([np.cos(cangle), np.sin(cangle)])
                )

            order_parameter_S, Q_tensor = (
                CalculateOrderParameter.calculate_order_parameter(
                    neighborhood_local_directors
                )
            )

            nodedata[f"{self._save_property_prefix}_nematic_order_parameter_S"] = (
                order_parameter_S
            )
            nodedata[f"{self._save_property_prefix}_Q_tensor"] = Q_tensor

        return BaseDataSetEntry(identifier=entry.identifier, data=current_graph)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--infile",
        required=True,
        type=str,
        help="Path to input file.",
    )
    parser.add_argument(
        "--mum_per_px",
        required=True,
        type=float,
        help="Microns per px.",
    )
    parser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="Path to output file.",
    )

    args = parser.parse_args()

    x = BaseDataSet.from_pickle(args.infile)
    x = BuildGraphTransform(args.mum_per_px)(x)
    x = CalculateCellNucleusShapeTransformation()(x)
    x = CalculateOrderParameter("cell_major_axis_angle_rad", "cell")(x)
    graph_nematic_ds = CalculateOrderParameter(
        "nucleus_major_axis_angle_rad", "nucleus"
    )(x)

    x.to_pickle(args.outfile)
