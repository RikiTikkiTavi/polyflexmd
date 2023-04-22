import matplotlib.pyplot as plt

import polyflexmd.data_analysis.data.types as types
import polyflexmd.data_analysis.transform.transform as transform
import pandas as pd


def plot_initial_system(system: types.LammpsSystemData) -> tuple[plt.Figure, plt.Axes]:
    fig: plt.Figure = plt.figure(figsize=(10, 10))
    ax: plt.Axes = fig.add_subplot(111, projection="3d")

    df_atoms_unfolded = transform.unfold_coordinates_df(system.atoms, system)

    df_atoms_unfolded_sample = df_atoms_unfolded.loc[df_atoms_unfolded["molecule-ID"] == 1]

    dims = ["x", "y", "z"]

    ax.scatter(df_atoms_unfolded_sample["x"], df_atoms_unfolded_sample["y"], df_atoms_unfolded_sample["z"], s=100, ec="w")

    ax.plot(df_atoms_unfolded_sample["x"], df_atoms_unfolded_sample["y"], df_atoms_unfolded_sample["z"], color="tab:gray")

    return fig, ax
