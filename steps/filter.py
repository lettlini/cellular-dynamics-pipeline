from argparse import ArgumentParser

from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.datasets.image import ImageDataset
from core_data_utils.transformations import BaseFilter


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

    x.to_pickle(args.outfile)
