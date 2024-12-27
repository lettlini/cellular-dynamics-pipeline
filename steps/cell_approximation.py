import pickle
from argparse import ArgumentParser

import cv2
import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.datasets.image import ImageDataset
from core_data_utils.transformations import BaseDataSetTransformation
from nuclei_segmentation import get_disconnected


class CellApproximationTransform(BaseDataSetTransformation):
    def __init__(self, cell_cutoff_px: int):
        self._cell_cutoff_px = cell_cutoff_px

        super().__init__()

    def _transform_single_entry(self, entry: BaseDataSetEntry) -> BaseDataSetEntry:
        """Function to convert nuclei brightfield microscopy images (grayscale) to cell masks"""

        image = entry.data

        binary_nuclei_mask: np.array = (image > 0).astype(np.int8)
        _, label_image = cv2.connectedComponents(binary_nuclei_mask)

        dislabels = label_image.copy().astype(np.int32)
        bg_mask = np.ones_like(label_image)
        inimage = cv2.distanceTransform(
            (label_image == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

        if self._cell_cutoff_px is not None:
            bg_mask[inimage >= self._cell_cutoff_px] = 0

        inimage = cv2.merge(3 * [inimage]).astype(np.uint8)

        dislabels = cv2.watershed(inimage, dislabels)

        # step 4: some post-processing
        dislabels[dislabels == -1] = 0  # set boundaries to 0
        dislabels[get_disconnected(dislabels) == 1] = 0

        if self._cell_cutoff_px is not None:
            dislabels[bg_mask == 0] = 0

        dislabels = self._ensure_cell_integrity(dislabels)

        dislabels = (dislabels > 0).astype(np.int8)

        return BaseDataSetEntry(identifier=entry.identifier, data=dislabels)

    def _ensure_cell_integrity(self, label_image: np.array) -> np.array:
        all_labels = np.setdiff1d(np.unique(label_image), np.array([0]))
        set_zero_mask = np.zeros_like(label_image, dtype=bool)

        for label in all_labels:
            current_mask = (label_image == label).astype(np.int8)
            num_labels, lbim, stats, _ = cv2.connectedComponentsWithStats(
                current_mask, connectivity=8
            )

            if num_labels > 2:  # first label: bg, second label: 1st cc
                # we have too many connected components, set all to 0 exepct largest one
                keep_label = np.argmax(stats[1:, 4]) + 1
                set_zero_mask = np.logical_or(
                    set_zero_mask, np.logical_and(lbim != keep_label, current_mask > 0)
                )

        ret_image = label_image.copy()
        ret_image[set_zero_mask] = 0

        return ret_image

    def _post_processing(self, data_dict: dict):
        return ImageDataset(data_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--infile", required=True, type=str, help="Absolute path to input file"
    )
    parser.add_argument(
        "--outfile", required=True, type=str, help="Path to output file"
    )
    parser.add_argument(
        "--cell_cutoff_px",
        required=True,
        type=float,
        help="Maximum radius of individual cells",
    )

    args = parser.parse_args()

    x = BaseDataSet.from_pickle(args.infile)

    x = CellApproximationTransform(cell_cutoff_px=args.cell_cutoff_px)(dataset=x)

    x.to_pickle(args.outfile)
