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
        dimension: typing.Optional[str] = None,
        title: typing.Optional[str] = None,
        ax: typing.Optional[plt.Axes] = None,
        label: typing.Optional[str] = None,
        color: typing.Optional[str] = None,
        marker: typing.Optional[matplotlib.markers.MarkerStyle] = None,
) -> plt.Axes:
    df = df_msd.copy()
    dR_col = "$ \sqrt {{\langle (\Delta R(t))^2 \\rangle}} / L$"

    if dimension is not None:
        dR_col = f"$ \sqrt {{\langle (\Delta R_{dimension}(t))^2 \\rangle}} / L$"

    df[dR_col] = np.sqrt(df[col]) / L_contour

    ax: plt.Axes = sns.scatterplot(
        df,
        x="t/LJ",
        y=dR_col,
        ax=ax,
        label=label,
        color=color,
        marker=marker,
        linewidths=1,
        edgecolors="black",
        s=10
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

    ax.set(title=title)

    return ax
