import os
from argparse import ArgumentParser

import cv2
import toml
from core_data_utils.datasets import BaseDataSet, BaseDataSetEntry
from core_data_utils.datasets.image import ImageDataset


def load_dir_eliane(pdir) -> BaseDataSet:
    ds = ImageDataset.from_directory(pdir)

    return ds


def load_dir_juergen(pdir) -> BaseDataSet:

    filenames = [
        fn for fn in os.listdir(pdir) if os.path.isfile(os.path.join(pdir, fn))
    ]
    data = {}

    for fpath in filenames:
        if fpath.endswith("c2.png"):
            image = cv2.cvtColor(
                cv2.imread(os.path.join(pdir, fpath), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            data[fpath] = BaseDataSetEntry(identifier=fpath, data=image, metadata={})
    return BaseDataSet(dataset_entries=data)


if __name__ == "__main__":

    cv2.setNumThreads(0)

    parser = ArgumentParser()
    parser.add_argument("--indir", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument(
        "--cpus",
        required=True,
        type=int,
        help="CPU cores to use.",
    )

    args = parser.parse_args()

    dataset_config = toml.load(args.dataset_config)

    provider = dataset_config["experimental-parameters"]["provider"]

    if provider.lower() == "eliane":
        x = load_dir_eliane(args.indir)

    elif provider.lower() == "juergen":
        x = load_dir_juergen(args.indir)

    else:
        raise RuntimeError(f"Data provider '{provider}' unknown.")

    x.to_pickle(args.outfile)
