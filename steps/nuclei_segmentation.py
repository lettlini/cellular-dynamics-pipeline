import os
from argparse import ArgumentParser

import cv2
import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.datasets.image import ImageDataset
from core_data_utils.transformations import BaseDataSetTransformation
from stardist.models import StarDist2D

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # prevent stardist / tensorflow from complaining
)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_disconnected(timage: np.array) -> np.array:
    """Disconnects touching regions with different labels (stardist)"""

    dmask = np.zeros_like(timage)

    # get list of labels
    lls = np.unique(timage)
    lls = np.setdiff1d(lls, (0,))

    for current_label in lls:
        current_small_image = (timage == current_label).astype(np.uint8)
        current_small_image = cv2.erode(
            current_small_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        dmask += current_small_image

    return ~(dmask.astype(bool))


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


class StarDistSegmentationTransform(BaseDataSetTransformation):
    def __init__(
        self,
        prob_threshold: float | None = None,
    ):
        self._probability_threshold: float = prob_threshold
        self._stardist_model = StarDist2D.from_pretrained("2D_versatile_fluo")

        super().__init__()

    def _transform_single_entry(
        self, entry: BaseDataSetEntry, dataset_properties: dict
    ) -> BaseDataSetEntry:

        image = entry.data

        labels, _ = self._stardist_model.predict_instances(
            image, prob_thresh=self._probability_threshold
        )

        # We need to disconnect touching labels:
        dmask = get_disconnected(labels)
        labels[dmask == 1] = 0

        # convert label image to binary mask
        labels = (labels > 0).astype(np.int8)

        return BaseDataSetEntry(identifier=entry.identifier, data=labels)


class RemoveSmallObjectsTransform(BaseDataSetTransformation):
    def __init__(
        self,
        min_nuc_area_px2: float,
    ) -> None:
        self._min_nuclei_area_px = min_nuc_area_px2

        super().__init__()

    def _transform_single_entry(
        self, entry: BaseDataSetEntry, dataset_properties: dict
    ) -> BaseDataSetEntry:

        image = entry.data

        image = (image > 0).astype(np.int8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image)

        for component_idx in range(1, num_labels):
            if stats[component_idx, 4] < self._min_nuclei_area_px:
                labels[labels == component_idx] = 0

        # convert label image to binary mask
        labels = (labels > 0).astype(np.int8)

        return BaseDataSetEntry(identifier=entry.identifier, data=labels)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--infile", required=True, type=str, help="Absolute path to input file"
    )
    parser.add_argument(
        "--outfile", required=True, type=str, help="Path to output file"
    )
    parser.add_argument(
        "--stardist_probility_threshold",
        required=True,
        type=float,
        help="Stardist probability threshold",
    )
    parser.add_argument(
        "--min_nucleus_area_pxsq",
        required=True,
        type=float,
        help="Nuclei smaller than 'min_nucleus_area_mumsq' will be removed",
    )
    parser.add_argument(
        "--cpus",
        required=True,
        type=int,
        help="CPU cores to use.",
    )

    args = parser.parse_args()

    cv2.setNumThreads(args.cpus)

    x = BaseDataSet.from_pickle(args.infile)

    x = GrayScaleTransform()(x)
    x = MinMaxScaleTransform()(x)

    x = StarDistSegmentationTransform(prob_threshold=args.stardist_probility_threshold)(
        dataset=x
    )
    x = RemoveSmallObjectsTransform(args.min_nucleus_area_pxsq)(dataset=x)

    x.to_pickle(args.outfile)
