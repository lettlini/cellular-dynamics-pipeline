from argparse import ArgumentParser

import polars as pl
import toml


def get_dataset_metadata(dataset_name: str, provider_name: str) -> tuple[str, str]:

    if provider_name.lower() == "eliane":
        cell_line = None
        if "hela" in dataset_name.lower():
            cell_line = "HeLa"
        elif "caski" in dataset_name.lower():
            cell_line = "CaSki"
        elif "ms751" in dataset_name.lower():
            cell_line = "MS751"
        else:
            raise RuntimeError(
                f"Could not infer cell line from dataset name '{dataset_name}'"
            )

        if dataset_name.lower().endswith("_cc"):
            cell_culture_method = "co-culture"
        else:
            cell_culture_method = "control"

        return cell_line, cell_culture_method

    if provider_name.lower() == "juergen":
        return "MCF-10A", "control"

    raise ValueError(f"Unknown provider {provider_name}")


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

    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--basename", type=str, required=True)

    parser.add_argument(
        "--cpus",
        required=True,
        type=int,
        help="CPU cores to use.",
    )

    args = parser.parse_args()

    cell_tracking_df = pl.read_ipc(args.infile, memory_map=False)
    dataset_config = toml.load(args.dataset_config)
    provider = dataset_config["experimental-parameters"]["provider"]

    cell_line_name, cell_culture_methodology = get_dataset_metadata(
        args.basename, provider_name=provider
    )

    # create 'cell_line_name' and 'cell_culture_methodology' columns in df
    cell_tracking_df = cell_tracking_df.with_columns(
        pl.lit(cell_line_name).alias("cell_line_name")
    )
    cell_tracking_df = cell_tracking_df.with_columns(
        pl.lit(cell_culture_methodology).alias("cell_culture_methodology")
    )
    cell_tracking_df = cell_tracking_df.with_columns(
        pl.lit(provider.lower()).alias("dataset_provider")
    )

    cell_tracking_df.write_ipc(args.outfile, compression="lz4")
