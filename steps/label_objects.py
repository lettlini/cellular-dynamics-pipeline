from argparse import ArgumentParser

import cv2
import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.transformations import BaseDataSetTransformation


class LabelImagesTransformation(BaseDataSetTransformation):
    def _transform_single_entry(self, entry: BaseDataSetEntry) -> BaseDataSetEntry:
        image = entry.data

        image = (image > 0).astype(np.uint8)

        _, labelim = cv2.connectedComponents(image)

        return BaseDataSetEntry(identifier=entry.identifier, data=labelim)


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

    args = parser.parse_args()

    x = BaseDataSet.from_pickle(args.infile)
    x = LabelImagesTransformation()(x)
    x.to_pickle(args.outfile)
