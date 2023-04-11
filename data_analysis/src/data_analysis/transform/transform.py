import numpy as np
import pandas as pd
import enum
import data_analysis.data.types
import pathlib
import functools


class AtomGroup(enum.Enum):
    ROOT = 1
    FREE = 2
    LEAF = 3


def unfold_coordinate(val: float, i: float, box_length: float):
    return val + i * box_length


def calculate_end_to_end(molecule_traj_step_df: pd.DataFrame, system_data: data_analysis.data.types.LammpsSystemData):
    root_atom_data = molecule_traj_step_df \
        .loc[molecule_traj_step_df["type"] == AtomGroup.ROOT.value] \
        .sort_values("id") \
        .iloc[0]

    leaf_atom_data = molecule_traj_step_df \
        .loc[molecule_traj_step_df["type"] == AtomGroup.LEAF.value] \
        .sort_values("id", ascending=False) \
        .iloc[0]

    root_coordinates_unfolded = np.zeros(3)
    leaf_coordinates_unfolded = np.zeros(3)

    for dim_i, dim_name in enumerate(('x', 'y', 'z')):
        root_coordinates_unfolded[dim_i] = unfold_coordinate(
            val=root_atom_data[dim_name],
            i=root_atom_data[f"i{dim_name}"],
            box_length=system_data.box.bounds[dim_i][1] - system_data.box.bounds[dim_i][0]
        )

        leaf_coordinates_unfolded[dim_i] = unfold_coordinate(
            val=leaf_atom_data[dim_name],
            i=leaf_atom_data[f"i{dim_name}"],
            box_length=system_data.box.bounds[dim_i][1] - system_data.box.bounds[dim_i][0]
        )

    return np.linalg.norm(leaf_coordinates_unfolded - root_coordinates_unfolded)


def join_raw_trajectory_df_with_system_data(
        raw_trajectory_df: pd.DataFrame,
        system_data: data_analysis.data.types.LammpsSystemData
) -> pd.DataFrame:
    return raw_trajectory_df.join(
        system_data.atoms["molecule-ID"],
        on="id"
    )


def calc_end_to_end_df(
        trajectory_df: pd.DataFrame,
        system_data: data_analysis.data.types.LammpsSystemData
) -> pd.DataFrame:
    return trajectory_df.groupby(["molecule-ID", "t"]).apply(
        functools.partial(calculate_end_to_end, system_data=system_data)
    )
