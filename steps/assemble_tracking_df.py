from argparse import ArgumentParser
from collections import defaultdict
from typing import Iterable, Optional

import polars as pl
from core_data_utils.datasets import BaseDataSet
from tqdm import trange


class CellTrackAssembler:

    def __init__(
        self,
        delta_t_minutes: float,
        include_attr: Optional[Iterable[str]] = None,
        exclude_attr: Optional[Iterable[str]] = None,
    ) -> None:
        self._seen_node_ids = defaultdict(list)
        self._current_track_id: int = 1

        if (include_attr is not None) and (exclude_attr is not None):
            raise ValueError("Cannot specify both 'include_attrs' and 'exclude_attr'.")

        self._delta_t_minutes = delta_t_minutes
        self._attributes_to_include = include_attr
        self._attributes_to_exclude = exclude_attr

    def _extract_properties(self, properties: dict) -> dict:
        """
        Function for extracting properties according to
            'self._attributes_to_include' and 'self._attributes_to_exclude'
        Args:
            properties (dict): Node property dictionary from which attributes
                should be extracted
        Returns:
            (dict): dictionary containing the relevant properties as specified
                via include_attrs and exclude_attrs
        """

        if self._attributes_to_include is not None:
            return {
                k: v for k, v in properties.items() if k in self._attributes_to_include
            }
        if self._attributes_to_exclude is not None:
            return {
                k: v
                for k, v in properties.items()
                if k not in self._attributes_to_exclude
            }
        # both are None, that means return everything!
        return properties

    def _get_node_information(
        self,
        graph_dataset: BaseDataSet,
        cindex: int,
        node_label: int,
        track_time_index: int,
    ) -> dict:
        cgraph = graph_dataset[cindex].data
        cprops = cgraph.nodes[node_label]

        df_columns = self._extract_properties(cprops)

        df_columns["track_id"] = self._current_track_id
        df_columns["object_id"] = node_label
        df_columns["image_id"] = graph_dataset[cindex].identifier
        df_columns["frame_id"] = cindex
        df_columns["track_relative_time_minutes"] = (
            track_time_index * self._delta_t_minutes
        )

        next_object_id = (
            cprops["next_object_id"] if "next_object_id" in cprops else None
        )
        self._seen_node_ids[cindex].append(node_label)

        return df_columns, next_object_id

    def _track_single_node(
        self, tds: BaseDataSet, sindex: int, node_label: int
    ) -> list[dict]:
        new_df_rows: list[dict] = []

        assert node_label not in self._seen_node_ids[sindex]

        track_time_index = 0
        current_props, nobj_id = self._get_node_information(
            tds, sindex, node_label=node_label, track_time_index=track_time_index
        )
        current_props["starting_frame"] = sindex
        new_df_rows.append(current_props)

        while (nobj_id is not None) and (sindex + track_time_index + 1) < len(tds):
            track_time_index += 1
            current_props, nobj_id = self._get_node_information(
                tds,
                sindex + track_time_index,
                node_label=nobj_id,
                track_time_index=track_time_index,
            )
            current_props["starting_frame"] = sindex
            new_df_rows.append(current_props)

        self._current_track_id += 1
        return new_df_rows

    def __call__(self, tracked_cell_graph_ds: BaseDataSet) -> pl.DataFrame:

        df_rows: list[dict] = []

        for sindex in trange(len(tracked_cell_graph_ds)):
            current_graph = tracked_cell_graph_ds[sindex].data
            for nodeid in current_graph.nodes:
                if nodeid not in self._seen_node_ids[sindex]:
                    nrows = self._track_single_node(
                        tracked_cell_graph_ds, sindex, nodeid
                    )

                    assert len(nrows) >= 1  # at least return current information

                    df_rows.extend(nrows)
        return pl.DataFrame(df_rows)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--infile",
        required=True,
        type=str,
        help="Path to file containing graph dataset with tracking information.",
    )
    parser.add_argument(
        "--outfile",
        required=True,
        type=str,
        help="Path to write tracking dataframe to.",
    )
    parser.add_argument(
        "--delta_t_minutes",
        required=True,
        type=float,
        help="Time between two adjacent frames (in minutes).",
    )

    parser.add_argument("--exclude_attrs", type=str, required=True)
    parser.add_argument("--include_attrs", type=str, required=True)

    parser.add_argument(
        "--cpus",
        required=True,
        type=int,
        help="CPU cores to use.",
    )

    args = parser.parse_args()

    x = BaseDataSet.from_pickle(args.infile)

    exclude_attrs = args.exclude_attrs.split(",") if args.exclude_attrs != "" else None
    include_attrs = args.include_attrs.split(",") if args.include_attrs != "" else None

    cell_tracking_df = CellTrackAssembler(
        args.delta_t_minutes,
        include_attr=include_attrs,
        exclude_attr=exclude_attrs,
    )(x)

    cell_tracking_df.write_ipc(args.outfile, compression="lz4")
