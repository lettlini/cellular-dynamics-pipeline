import os
from argparse import ArgumentParser

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Fira Sans"],
    }
)


def plot_d2min_vs_crsd(crsd, d2min, title: str):

    f = plt.figure(figsize=(5, 5))
    ax = f.add_subplot(111)

    kde = gaussian_kde(np.vstack((crsd, d2min)))
    z = kde(np.vstack((crsd, d2min)))

    # Sort the points by density, so that the densest points are plotted last
    idx = np.argsort(z)
    crsd, d2min, z = crsd[idx], d2min[idx], z[idx]

    ax.scatter(crsd, d2min, c=z, s=1, edgecolor=None, cmap="Reds", alpha=0.5)
    ax.plot(
        [0, np.percentile(crsd, 97)],
        [0, np.percentile(crsd, 97)],
        "r--",
        alpha=0.25,
        linewidth=2,
    )

    ax.set_xlim(0, np.percentile(crsd, 97))
    ax.set_ylim(0, np.percentile(d2min, 97))

    ax.set_xlabel(r"CRSD in $\left[\mu m^2\right]$", fontfamily="sans-serif")
    ax.set_ylabel(
        r"$D^2 _\text{min}$ in $\left[\mu m^2\right]$", fontfamily="sans-serif"
    )
    ax.set_title(
        title,
        fontweight="bold",
        fontfamily="sans-serif",
        pad=20,
    )

    sns.despine(f, ax)
    f.tight_layout()

    return f


def d2min_vs_crsd(df_file: str, parent_dir_out: str):
    # read in combined cell tracking dataframe from ipc file
    big_dataframe: pl.DataFrame = pl.read_ipc(df_file, memory_map=False)

    # get a list of all lag times
    all_lag_times = [
        lt.removesuffix("_minutes").removeprefix("D2min_")
        for lt in [c for c in big_dataframe.columns if "D2min_" in c]
    ]

    for lt in all_lag_times:
        crsd_col = f"cage_relative_squared_displacement_mum_squared_{lt}_min"
        d2min_col = f"D2min_{lt}_minutes"

        current_df = big_dataframe.filter(
            pl.col(crsd_col).is_not_nan() & pl.col(d2min_col).is_not_nan()
        )

        # calculate pearson correlation coefficient
        pearson_corr = pearsonr(
            current_df[crsd_col].to_numpy(), current_df[d2min_col].to_numpy()
        ).statistic

        current_sampled_df = current_df.sample(n=min(30_000, len(current_df)))

        current_figure = plot_d2min_vs_crsd(
            current_sampled_df[crsd_col],
            current_sampled_df[d2min_col],
            r"$D^2 _\text{min}$ vs CRSD ($\tau = "
            + lt
            + " \textit{{min}}$)"
            + f"(Pearson correlation: {pearson_corr:.2f})",
        )

        current_figure.savefig(
            os.path.join(parent_dir_out, f"d2min_vs_crsd_{lt}_minutes.png"),
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataframe_file", type=str, required=True)
    parser.add_argument("--parent_dir_out", type=str, required=True)
    args = parser.parse_args()

    d2min_vs_crsd(args.dataframe_file, args.parent_dir_out)
