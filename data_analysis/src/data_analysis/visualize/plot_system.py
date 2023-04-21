from typing import Optional

import matplotlib.pyplot as plt

import data_analysis.data.types
import data_analysis.data.read as read
import data_analysis.transform.transform as transform

import numpy as np


def plot_fene_system(
        system: data_analysis.data.types.LammpsSystemData,
        n_mol: int = 3,
        fig: Optional[plt.Figure] = None,
        axs: Optional[list[plt.Axes]] = None
):
    df_atoms = transform.unfold_coordinates_df(system.atoms, system)
    mol_sample = np.random.choice(df_atoms["molecule-ID"].unique(), n_mol, replace=False)

    if fig is None:
        fig = plt.figure(figsize=(10, 20))

    if axs is None:
        axs = [fig.add_subplot(3, 1, i, projection="3d") for i in range(1, n_mol + 1)]

    for ax, (mol_id, df_mol) in zip(axs, df_atoms.loc[df_atoms["molecule-ID"].isin(mol_sample)].groupby("molecule-ID")):
        ax.scatter(df_mol["x"], df_mol["y"], df_mol["z"], ec="w", s=100)
        ax.plot(df_mol["x"], df_mol["y"], df_mol["z"], color="tab:gray")

    return fig, axs
