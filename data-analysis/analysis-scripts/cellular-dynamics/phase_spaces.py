import os
import sys
from argparse import ArgumentParser
from itertools import product
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Fira Sans"],
        "mathtext.fontset": "stixsans",
    }
)

latex_strings = {
    "cell_shape": {"name": "Cell Shape", "formula": "cs", "unit": ""},
    "cell_area_mum_squared": {
        "name": "Cell Area",
        "formula": "A_C",
        "unit": "\\mu m^2",
    },
    "local_density_per_mum_squared": {
        "name": "Local Density",
        "formula": "\\rho",
        "unit": "\\mu m^{-2}",
    },
    "nucleus_shape": {"name": "Nucleus Shape", "formula": "ns", "unit": ""},
    "nucleus_area_mum_squared": {
        "name": "Nucleus Area",
        "formula": "A_N",
        "unit": "\\mu m^2",
    },
    "D2min": {
        "name": "D^2{_\\text{min}}",
        "formula": "D^2{_\\text{min}}",
        "unit": "\\mu m^2",
    },
    "cage_relative_squared_displacement_mum_squared": {
        "name": "\\text{CRSD}",
        "formula": "\\text{CRSD}",
        "unit": "\\mu m^2",
    },
}


def get_label_string(quant_name, exponent) -> str:

    y_unit = f"{latex_strings[quant_name]['unit']}"

    if len(y_unit) > 0:
        y_unit = (
            "\\left[" + y_unit + "\\right]"
            if exponent == 0
            else "\\left[ 10^{" + str(exponent) + "} " + y_unit + " \\right]"
        )
        y_unit = f"in {y_unit}"
    else:
        y_unit = "10^{" + str(exponent) + "}" if exponent != 0 else ""

    label = (
        latex_strings[quant_name]["name"]
        + "\\left( "
        + latex_strings[quant_name]["formula"]
        + " \\right) "
        + y_unit
    )

    return label


def rescale_data(data: np.array) -> np.array:
    data = data.copy()
    exponent: int = 0

    while np.abs(data).max() < 1:
        data *= 10.0
        exponent += 1

    return data, -exponent


def create_2d_heatmap(
    independent_x, indenpendent_y, dependent_z, num_bins=20, min_count: int = 2
) -> tuple[np.array, tuple[np.array, np.array]]:

    # Create 2D histogram-like bins
    x_bins = np.linspace(min(independent_x), max(independent_x), num_bins + 1)
    y_bins = np.linspace(min(indenpendent_y), max(indenpendent_y), num_bins + 1)

    # Digitize the x and y data
    x_indices = np.digitize(independent_x, x_bins) - 1
    y_indices = np.digitize(indenpendent_y, y_bins) - 1

    # Create empty grid for the values
    heatmap_data = np.zeros((num_bins, num_bins))
    count_grid = np.zeros((num_bins, num_bins))

    # Fill the grid with mean values
    for x_idx, y_idx, val in zip(x_indices, y_indices, dependent_z):
        if x_idx < num_bins and y_idx < num_bins:  # Prevent index out of bounds
            heatmap_data[y_idx, x_idx] += val
            count_grid[y_idx, x_idx] += 1

    # Calculate means (avoid division by zero)
    mask = count_grid > min_count
    heatmap_data[mask] = heatmap_data[mask] / count_grid[mask]
    heatmap_data[~mask] = np.nan

    # the origin should be in the lower left corner
    heatmap_data = np.flipud(heatmap_data)
    y_bins = np.flipud(y_bins)

    return heatmap_data, (x_bins, y_bins)


def plot_heatmap(
    matrix,
    x_bins,
    y_bins,
    x_label,
    y_label,
    title,
    colorbar_label,
    z_cutoff: Optional[float] = None,
):
    # Create the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if z_cutoff:
        if np.nanmax(matrix) > z_cutoff:
            ticks = np.linspace(np.nanmin(matrix), z_cutoff, 6)
            labels = [*map(lambda x: f"{x:.2f}", ticks[:-1])] + [
                f"$\geq$ {z_cutoff:.1f}"
            ]
        else:
            ticks = np.linspace(np.nanmin(matrix), np.nanmax(matrix), 6)
            labels = [*map(lambda x: f"{x:.2f}", ticks)]
    else:
        ticks = np.linspace(np.nanmin(matrix), np.nanmax(matrix), 6)
        labels = [*map(lambda x: f"{x:.2f}", ticks)]

    matrix = np.clip(matrix, 0, z_cutoff) if z_cutoff else matrix

    im = ax.imshow(matrix, cmap="Reds", interpolation="nearest", origin="upper")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label(colorbar_label)

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

    ax.set_xticks(range(len(x_bins)), np.round(x_bins, 2))
    ax.set_yticks(range(len(y_bins)), np.round(y_bins, 2))

    sns.despine(fig, ax)

    # Customize the plot
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_title(label=title)

    # Rotate tick labels
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45, ha="right")

    return fig


def phase_spaces(df_file: str, parent_dir_out: str):

    # step 1: read in combined cell tracking dataframe from ipc file
    big_dataframe: pl.DataFrame = pl.read_ipc(df_file, memory_map=False)

    # get a list of all lag times
    all_lag_times = [
        lt.removesuffix("_minutes").removeprefix("D2min_")
        for lt in [c for c in big_dataframe.columns if "D2min_" in c]
    ]

    all_motility_measures = ["D2min", "cage_relative_squared_displacement_mum_squared"]

    # specify phase_space independent variables, x,y tuples
    indendent_vars: list[tuple[str, str]] = [
        ("cell_shape", "cell_area_mum_squared"),
        ("cell_shape", "local_density_per_mum_squared"),
        ("nucleus_shape", "nucleus_area_mum_squared"),
    ]

    for (inx_col, iny_col), mot_m, lag_time in product(
        indendent_vars, all_motility_measures, all_lag_times
    ):

        motility_measure = (
            f"{mot_m}_{lag_time}_minutes"
            if mot_m == "D2min"
            else f"{mot_m}_{lag_time}_min"
        )

        current_filtered_df: pl.DataFrame = big_dataframe.filter(
            pl.col(motility_measure).is_not_nan()
        )

        x_vals = current_filtered_df[inx_col].to_numpy()
        y_vals = current_filtered_df[iny_col].to_numpy()

        x_vals, x_exponent = rescale_data(x_vals)
        y_vals, y_exponent = rescale_data(y_vals)

        x_label = get_label_string(inx_col, x_exponent)
        y_label = get_label_string(iny_col, y_exponent)
        matrix, (xbin, ybin) = create_2d_heatmap(
            x_vals,
            y_vals,
            current_filtered_df[motility_measure],
        )

        motility_str = (
            "$"
            + latex_strings[mot_m]["name"]
            + "$"
            + "$\\left(\\tau = "
            + lag_time
            + " min\\right)$"
        )

        fig = plot_heatmap(
            matrix,
            xbin,
            ybin,
            x_label=f"${x_label}$",
            y_label=f"${y_label}$",
            title=f"{motility_str} vs {latex_strings[inx_col]['name']} and {latex_strings[iny_col]['name']}",
            colorbar_label=motility_str,
            z_cutoff=np.percentile(
                current_filtered_df[motility_measure].to_numpy(), 96
            ),
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                parent_dir_out,
                f"{motility_measure}_vs_{inx_col}_and_{iny_col}.png",
            ),
            bbox_inches="tight",
            dpi=500,
        )

        plt.close(fig)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataframe_file", type=str, required=True)
    parser.add_argument("--parent_dir_out", type=str, required=True)
    args = parser.parse_args()

    phase_spaces(args.dataframe_file, args.parent_dir_out)
