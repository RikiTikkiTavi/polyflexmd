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


def unfold_coordinates_row(traj_row: pd.Series, system_data: data_analysis.data.types.LammpsSystemData) -> pd.Series:
    dimensions = ('x', 'y', 'z')
    coordinates = []

    for dim_i, dim_name in enumerate(dimensions):
        coordinates.append(unfold_coordinate(
            val=traj_row.loc[dim_name],
            i=traj_row.loc[f"i{dim_name}"],
            box_length=system_data.box.bounds[dim_i][1] - system_data.box.bounds[dim_i][0]
        ))

    return pd.Series(data=coordinates, index=dimensions)


def unfold_coordinates_df(
        trajectory_df: pd.DataFrame,
        system_data: data_analysis.data.types.LammpsSystemData
) -> pd.DataFrame:
    trajectory_df_unfolded = trajectory_df.copy()
    dimensions = ('x', 'y', 'z')

    for dim_i, dim_name in enumerate(dimensions):
        box_length = system_data.box.bounds[dim_i][1] - system_data.box.bounds[dim_i][0]
        trajectory_df_unfolded[dim_name] = trajectory_df[dim_name] + trajectory_df[f"i{dim_name}"] * box_length

    return trajectory_df_unfolded


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
    ).rename("R")


def calculate_ete_change_ens_avg(df_ete_t: pd.Series, df_ete_t_0: pd.Series) -> float:
    return ((df_ete_t - df_ete_t_0) ** 2).mean()


def calculate_ete_change_ens_avg_df(df_ete: pd.Series) -> pd.DataFrame:
    t_min = df_ete.index.get_level_values("t").min()
    ete_df_t_0 = df_ete.loc[:, t_min]

    return df_ete \
        .groupby(level="t") \
        .apply(functools.partial(calculate_ete_change_ens_avg, df_ete_t_0=ete_df_t_0)) \
        .rename("<R(t)-R(0)>")


def calculate_neigh_distance_avg_df(trajectory_df_unfolded: pd.DataFrame) -> float:
    t_max = trajectory_df_unfolded["t"].max()
    df_t_max = trajectory_df_unfolded.loc[trajectory_df_unfolded["t"] == t_max]

    return np.sum([(df_t_max[d].iloc[1:] - df_t_max[d].iloc[:-2]) ** 2 for d in ('x', 'y', 'z')], axis=1).mean()
