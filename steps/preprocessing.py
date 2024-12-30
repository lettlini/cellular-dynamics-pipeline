import pickle
from argparse import ArgumentParser
from typing import Any

import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.datasets.image import ImageDataset
from core_data_utils.transformations import BaseDataSetTransformation


class MinMaxScaleTransform(BaseDataSetTransformation):
    def _transform_single_entry(
        self, entry: BaseDataSetEntry, dataset_properties: dict
    ) -> BaseDataSetEntry:

        new_image = entry.data

        minval, maxval = np.min(new_image), np.max(new_image)
        new_image = (new_image - minval) / (maxval - minval)

        return BaseDataSetEntry(identifier=entry.identifier, data=new_image)


class GrayScaleTransform(BaseDataSetTransformation):
    def _transform_single_entry(
        self, entry: BaseDataSetEntry, dataset_properties: dict
    ) -> BaseDataSetEntry:

        image = entry.data

        new_image = image.mean(axis=2)
        return BaseDataSetEntry(identifier=entry.identifier, data=new_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--infile", required=True, type=str, help="Absolute path to input file"
    )
    parser.add_argument(
        "--outfile", required=True, type=str, help="Path to output file"
    )
    args = parser.parse_args()

    x = BaseDataSet.from_pickle(args.infile)

    x = GrayScaleTransform()(x)
    x = MinMaxScaleTransform()(x)

    x.to_pickle(args.outfile)
