import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # prevent stardist / tensorflow from complaining
)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from argparse import ArgumentParser
from typing import Any

import numpy as np

from core_data_utils.datasets.image import ImageDataset
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.transformations import BaseDataSetTransformation, BaseFilter


class MinMaxScaleTransform(BaseDataSetTransformation):
    def _transform_single_entry(self, entry: BaseDataSetEntry) -> BaseDataSetEntry:

        image = entry.data

        new_image = image.copy()
        minval, maxval = np.min(new_image), np.max(new_image)
        new_image = (new_image - minval) / (maxval - minval)

        return BaseDataSetEntry(identifier=entry.identifier, data=new_image)

    def _post_processing(self, data_dict: dict[str, Any]) -> Any:
        return ImageDataset(data_dict)


class GrayScaleTransform(BaseDataSetTransformation):

    def _transform_single_entry(self, entry: BaseDataSetEntry) -> BaseDataSetEntry:

        image = entry.data

        new_image = image.copy().mean(axis=2)
        return BaseDataSetEntry(identifier=entry.identifier, data=new_image)

    def _post_processing(self, data_dict: dict[str, Any]) -> Any:
        return ImageDataset(data_dict)


class FirstLastFilter(BaseFilter):

    def __init__(self, first_n: int, last_m: int) -> None:

        # set filter parameters
        self.first_n = first_n
        self.last_m = last_m

        super().__init__()

    def _filter_decision_single_entry(
        self, index: int, ds_entry: BaseDataSetEntry, **kwargs
    ) -> bool:
        if (index < self.first_n) or (index >= kwargs["dataset_length"] - self.last_m):
            return False
        return True

    def _global_dataset_properties(self, dataset: BaseDataSet) -> dict:
        return {"dataset_length": len(dataset)}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--infile", required=True, type=str, help="Absolute path to input file"
    )
    parser.add_argument(
        "--outfile", required=True, type=str, help="Path to output file"
    )
    parser.add_argument(
        "--drop_first_n",
        required=True,
        type=int,
        help="Drop first n entries from DataSet",
    )
    parser.add_argument(
        "--drop_last_m",
        required=True,
        type=int,
        help="Drop last m entries from DataSet",
    )
    args = parser.parse_args()

    x: ImageDataset = ImageDataset.from_directory(args.infile)
    x = FirstLastFilter(first_n=args.drop_first_n, last_m=args.drop_last_m)(x)
    x = GrayScaleTransform()(x)
    x = MinMaxScaleTransform()(x)

    x.to_pickle(args.outfile)
