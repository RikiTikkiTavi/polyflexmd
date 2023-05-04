from itertools import islice

import matplotlib.pyplot as plt
import matplotlib.colors

import polyflexmd.data_analysis.data.types as types
import polyflexmd.data_analysis.transform.transform as transform
import pandas as pd

import numpy as np

import numpy as np

import typing


def plot_cube(bounds: np.ndarray, ax: typing.Optional[plt.Axes] = None):
    xbounds, ybounds, zbounds = bounds

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])  # Set aspect ratio to 1:1:1

    # Determine the size of the cube based on the box bounds
    size = min(xbounds[1] - xbounds[0], ybounds[1] - ybounds[0], zbounds[1] - zbounds[0])

    # Determine the center point of the cube
    center = [(xbounds[0] + xbounds[1]) / 2, (ybounds[0] + ybounds[1]) / 2, (zbounds[0] + zbounds[1]) / 2]

    # Define the vertices of the cube
    vertices = [[center[0] - size / 2, center[1] - size / 2, center[2] - size / 2],
                [center[0] + size / 2, center[1] - size / 2, center[2] - size / 2],
                [center[0] + size / 2, center[1] + size / 2, center[2] - size / 2],
                [center[0] - size / 2, center[1] + size / 2, center[2] - size / 2],
                [center[0] - size / 2, center[1] - size / 2, center[2] + size / 2],
                [center[0] + size / 2, center[1] - size / 2, center[2] + size / 2],
                [center[0] + size / 2, center[1] + size / 2, center[2] + size / 2],
                [center[0] - size / 2, center[1] + size / 2, center[2] + size / 2]]

    # Define the edges of the cube
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    # Plot the cube
    for edge in edges:
        x = [vertices[edge[0]][0], vertices[edge[1]][0]]
        y = [vertices[edge[0]][1], vertices[edge[1]][1]]
        z = [vertices[edge[0]][2], vertices[edge[1]][2]]
        ax.plot(x, y, z, color='black')

    return ax


def plot_initial_system(
        df_trajectory_unfolded: pd.DataFrame,
        system: types.LammpsSystemData,
        n_molecules: int,
        plot_box: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig: plt.Figure = plt.figure(figsize=(10, 10))
    ax: plt.Axes = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    # for i in range(n_molecules):
    #    axs.append(fig.add_subplot(n_molecules, 1, i))

    molecules_sample: np.ndarray = np.random.choice(
        a=df_trajectory_unfolded["molecule-ID"].unique(),
        size=n_molecules,
        replace=False
    )

    initial_state_sample: pd.DataFrame = df_trajectory_unfolded.loc[
        (df_trajectory_unfolded["t"] == 0) &
        (df_trajectory_unfolded["molecule-ID"].isin(molecules_sample))
        ]

    if plot_box:
        plot_cube(system.box.bounds, ax)

    colors = list(islice(matplotlib.colors.TABLEAU_COLORS, n_molecules))

    for color, (mol_id, df_mol) in zip(colors, initial_state_sample.groupby("molecule-ID")):
        ax.scatter(df_mol["x"].iloc[0], df_mol["y"].iloc[0], df_mol["z"].iloc[0], s=100, color=color, edgecolors="red", linewidth=6)
        ax.scatter(df_mol["x"], df_mol["y"], df_mol["z"], s=100, ec="w", label=f"molecule-ID={mol_id}")
        ax.scatter(df_mol["x"].iloc[-1], df_mol["y"].iloc[-1], df_mol["z"].iloc[-1], s=100, color=color, edgecolors="lime",
                   linewidth=6)
        ax.plot(
            df_mol["x"],
            df_mol["y"],
            df_mol["z"],
            color="tab:gray"
        )

    ax.legend()

    return fig, ax
