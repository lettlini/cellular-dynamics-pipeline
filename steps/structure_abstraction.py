from argparse import ArgumentParser
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.transformations import (
    BaseDataSetTransformation,
    BaseMultiDataSetTransformation,
)


class ObjectInformationTransform(BaseDataSetTransformation):
    def __init__(
        self,
        mum_px: float,
    ) -> None:

        self._mum_per_px: float = mum_px

        super().__init__()

    def _transform_single_entry(
        self, entry: BaseDataSetEntry, dataset_properties: dict
    ) -> tuple[str, dict]:

        image = entry.data  # image is already labelled
        binary_image = (image > 0).astype(np.uint8)

        object_properties: dict = {}

        # step 1: label image
        num_objects, labelim, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )

        # step 3: find contours in binary image
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not len(contours) == (num_objects - 1):
            raise RuntimeError("Number of contours != number of objects")

        for contour in contours:

            contour2labelim = np.unique(
                labelim[contour.squeeze()[:, 1], contour.squeeze()[:, 0]]
            ).item()

            original_label = np.unique(image[labelim == contour2labelim]).item()

            perimeter = cv2.arcLength(contour, closed=True)
            area = stats[contour2labelim, cv2.CC_STAT_AREA]

            (_, (minor_axis, major_axis), angle) = cv2.fitEllipse(contour)

            if original_label in object_properties:
                raise RuntimeError(
                    f"Object ID '{contour2labelim}' already present in object property dictionary."
                )

            object_properties[original_label] = {
                "area_mum_squared": area * (self._mum_per_px**2),
                "perimeter_mum": perimeter * self._mum_per_px,
                "major_axis_mum": major_axis * self._mum_per_px,
                "minor_axis_mum": minor_axis * self._mum_per_px,
                "major_axis_angle_rad": ((np.pi * angle) / 180.0) - np.pi / 2,
                "centroid_x": centroids[contour2labelim, 0],
                "centroid_y": centroids[contour2labelim, 1],
            }

        return BaseDataSetEntry(identifier=entry.identifier, data=object_properties)


class MergeCellNucleiInformation(BaseMultiDataSetTransformation):
    def _transform_single_entry(
        self, entry: BaseDataSetEntry, dataset_properties: dict
    ) -> BaseDataSetEntry:

        nuc_labelim = entry.data["nuclei_labels"]
        cell_labelim = entry.data["cell_labels"]
        nuc_props = entry.data["nuclei_properties"]
        cell_props = entry.data["cell_properties"]

        nuc_ids = np.setdiff1d(nuc_labelim, 0)
        cell_ids = np.setdiff1d(cell_labelim, 0)

        merged_properties = {}

        assert len(nuc_ids) == len(cell_ids)

        for nucleus_id in nuc_ids:
            ccids = np.setdiff1d(cell_labelim[nuc_labelim == nucleus_id], 0)
            assert len(ccids) == 1, f"{ccids}"
            cell_counterpart = ccids.item()

            assert (
                cell_counterpart not in merged_properties
            ), f"Cell with ID '{cell_counterpart}' already in merged properties dict"

            merged_properties[cell_counterpart] = {
                f"cell_{k}": v for k, v in cell_props[cell_counterpart].items()
            }
            merged_properties[cell_counterpart].update(
                {f"nucleus_{k}": v for k, v in nuc_props[nucleus_id].items()}
            )

        return BaseDataSetEntry(identifier=entry.identifier, data=merged_properties)

    def __call__(
        self, nuc_label_ds, cell_label_ds, nuc_prop_ds, cell_prop_ds
    ) -> BaseDataSet:
        return super()._transform(
            nuclei_labels=nuc_label_ds,
            cell_labels=cell_label_ds,
            nuclei_properties=nuc_prop_ds,
            cell_properties=cell_prop_ds,
        )


class IdentifyNeighborsTransformation(BaseMultiDataSetTransformation):
    def __init__(
        self, mum_px: float, cutout_size: int = 30, dilation_size: int = 10
    ) -> None:
        self._mum_per_px = mum_px
        self._cutout_size = cutout_size
        self._dilation_size = dilation_size

        super().__init__()

    def _transform_single_entry(
        self, entry: BaseDataSetEntry, dataset_properties: dict
    ) -> BaseDataSetEntry:
        props = entry.data["merged_properties"]
        labelim = entry.data["cell_labels"]

        cell_labels: np.array = np.setdiff1d(np.unique(labelim), 0)

        if len(cell_labels) != len(props):
            raise RuntimeError(
                "Unequal length of property dictionary and objects present in labelled image"
            )

        for current_cell_label in cell_labels:
            current_binary_mask = (labelim == current_cell_label).astype(np.uint8)
            contours, _ = cv2.findContours(
                current_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            neighbors = defaultdict(int)

            if len(contours) != 1:
                raise RuntimeError()

            contour = contours[0]

            # we need to use cutouts otherwise this is too slow!!
            empty_image = np.zeros_like(labelim, dtype=np.uint8)

            for point in contour:
                column, row = point[0]

                min_row: int = max(0, row - self._cutout_size)
                max_row: int = min(labelim.shape[0], row + self._cutout_size)
                min_col: int = max(0, column - self._cutout_size)
                max_col: int = min(labelim.shape[1], column + self._cutout_size)

                empty_image *= 0  # reset to 0 for each point
                empty_image[row, column] = 1
                empty_image_cutout = empty_image[min_row:max_row, min_col:max_col]
                labelim_cutout = labelim[min_row:max_row, min_col:max_col]

                dilated_empty_image_cutout = cv2.dilate(
                    empty_image_cutout,
                    cv2.getStructuringElement(
                        cv2.MORPH_RECT, (self._dilation_size, self._dilation_size)
                    ),
                )

                overlaps = np.setdiff1d(
                    labelim_cutout * dilated_empty_image_cutout, [0, current_cell_label]
                )

                for ov in overlaps:
                    neighbors[ov] += 1 * self._mum_per_px

            props[current_cell_label].update({"neighbors": neighbors})

        return BaseDataSetEntry(identifier=entry.identifier, data=props)

    def __call__(self, merged_properties, cell_labels, cpus: int = 1) -> Any:
        return super()._transform(
            cpus=cpus,
            merged_properties=merged_properties,
            cell_labels=cell_labels,
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--nuclei_infile",
        required=True,
        type=str,
        help="Absolute path to nuclei approximation dataset.",
    )
    parser.add_argument(
        "--cells_infile",
        required=True,
        type=str,
        help="Absolute path to cell approximation dataset.",
    )
    parser.add_argument(
        "--mum_per_px", required=True, type=float, help="Microns per px."
    )
    parser.add_argument(
        "--outfile", required=True, type=str, help="Path to output file"
    )
    parser.add_argument(
        "--cpus",
        required=True,
        type=int,
        help="CPU cores to use.",
    )

    args = parser.parse_args()

    # disable multi-threading in opencv
    cv2.setNumThreads(0)

    cells_labelled_ds = BaseDataSet.from_pickle(args.cells_infile)
    nuclei_labelled_ds = BaseDataSet.from_pickle(args.nuclei_infile)

    cell_properties = ObjectInformationTransform(args.mum_per_px)(
        cells_labelled_ds, cpus=args.cpus
    )
    nuclei_properties = ObjectInformationTransform(args.mum_per_px)(
        nuclei_labelled_ds, cpus=args.cpus
    )

    all_properties_merged = MergeCellNucleiInformation()(
        nuc_label_ds=nuclei_labelled_ds,
        cell_label_ds=cells_labelled_ds,
        nuc_prop_ds=nuclei_properties,
        cell_prop_ds=cell_properties,
    )

    all_properties_merged_neighbors = IdentifyNeighborsTransformation(
        mum_px=args.mum_per_px
    )(
        merged_properties=all_properties_merged,
        cell_labels=cells_labelled_ds,
        cpus=args.cpus,
    )

    all_properties_merged_neighbors.to_pickle(args.outfile)
