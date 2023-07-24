import typing

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import matplotlib.markers


def plot_MSD(
        df_msd: pd.DataFrame,
        log_scale: bool,
        l_K: float,
        L_contour: float,
        zeta: float,
        zeta_e: float,
        col: str = "dR^2",
        ci_alpha: float = 0.2,
        dimension: typing.Optional[str] = None,
        title: typing.Optional[str] = None,
        ax: typing.Optional[plt.Axes] = None,
        color: typing.Optional[str] = None,
        label: typing.Optional[str] = None,
        xlabel: typing.Optional[str] = None,
        ylabel: typing.Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    time_col = "t/LJ"

    col_delta = f"delta {col}"

    y = np.sqrt(df_msd[col]) / L_contour
    dy = df_msd[col_delta] / (np.sqrt(df_msd[col]) * L_contour * 2)

    if label is None:
        label = f"$l_K/L={l_K / L_contour : .2f}$"

    ax.plot(
        df_msd[time_col],
        y,
        c=color,
        label=label,
        # path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()]
    )
    ax.fill_between(
        x=df_msd[time_col],
        y1=y - dy,
        y2=y + dy,
        color=color,
        alpha=ci_alpha,
        linewidth=0
    )

    title_prefix = "MSD"

    if dimension is not None:
        title_prefix += f" in {dimension}-dimension"

    if log_scale:
        ax.set(xscale="log", yscale="log")
        title_prefix += " on log-log scale"

    zeta_title = f"$\zeta_e={zeta_e:.1f}$, $\zeta={zeta:.1f}$"

    if np.round(zeta, 1) == np.round(zeta_e, 1):
        zeta_title = f"$\zeta_e = \zeta= {zeta:.2f}$"

    if title is None:
        title = f"{title_prefix} for $l_K/L={l_K / L_contour:.2f}$, {zeta_title}, $L={L_contour}$"

    if xlabel is None:
        xlabel = time_col
    if ylabel is None:
        ylabel = "$ \sqrt {{\langle (\Delta R(t))^2 \\rangle}} / L$"

    ax.set(
        title=title,
        ylabel=ylabel,
        xlabel=xlabel
    )

    return ax
