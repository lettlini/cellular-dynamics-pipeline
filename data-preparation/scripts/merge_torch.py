from argparse import ArgumentParser

import torch
import pickle
from torch_geometric.data import Data
from core_data_utils.datasets import BaseDataSet


def merge_torch_datasets(individual_datasets: list[str]) -> list[Data]:
    """
    function for merging 'torch_geometric.data.Data' instances from different
    datasets into a single list of instances.

    Args:
        individual_datasets(list[str]):
    Returns:
        (list[Data]):
    """

    data_list: list[Data] = []
    all_ids = []

    for bds_path in individual_datasets:

        bds: BaseDataSet = BaseDataSet.from_pickle(bds_path)

        for entry in bds:
            assert entry.identifier not in all_ids
            all_ids.append(entry.identifier)

            entry.data["frame_id"] = entry.identifier

            data_list.append(entry.data)

    return data_list


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()

    with open(args.infile, "r", encoding="utf-8") as file:
        file_list = [line.strip() for line in file]

    all_torch_datases = merge_torch_datasets(file_list)

    with open(args.outfile, "wb") as of:
        pickle.dump(all_torch_datases, of)
