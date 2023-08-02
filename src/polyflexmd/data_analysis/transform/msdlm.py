import numpy as np
import pandas as pd
import dask.dataframe
import polyflexmd.data_analysis.transform.constants as constants
import polyflexmd.data_analysis.transform.msd as msd


def extract_lm(df_molecule_traj_step: pd.DataFrame) -> pd.Series:
    leaf_atom_data: pd.Series = df_molecule_traj_step \
        .loc[df_molecule_traj_step["type"] == constants.AtomGroup.LEAF.value] \
        .iloc[0]

    return leaf_atom_data[['x', 'y', 'z']]


def extract_lm_trajectory_df(
        df_trajectories: dask.dataframe.DataFrame,
        group_by_columns: list[str],
        time_col="t"
) -> dask.dataframe.DataFrame:
    """
    Extracts last monomer positions
    """
    gb = df_trajectories.groupby([time_col, *group_by_columns, "molecule-ID"])
    return gb.apply(extract_lm, meta=pd.DataFrame(columns=["x", "y", "z"]))


def calculate_msd_lm(df_lm_t: pd.DataFrame, df_lm_t_0: pd.DataFrame) -> pd.Series:
    dims = ["x", "y", "z"]

    df_lm_t_vec = df_lm_t[dims].to_numpy()
    df_lm_t_0_vec = df_lm_t_0[dims].to_numpy()

    msd_lm = np.sum((df_lm_t_vec - df_lm_t_0_vec) ** 2, axis=1)
    msd_lm_mean = msd_lm.mean()
    msd_lm_std_of_mean = msd_lm.std() / np.sqrt(msd_lm.shape[0])

    return pd.Series([msd_lm_mean, 3 * msd_lm_std_of_mean], ["dr_N^2", "delta dr_N^2"])


def calculate_msd_lm_df(df_lm_trajectory: pd.DataFrame, group_by_columns: list[str], time_col="t") -> pd.DataFrame:
    df_lm_trajectory = df_lm_trajectory.reset_index()
    dfs = []
    if len(group_by_columns) == 0:
        grouper = np.repeat(True, df_lm_trajectory.shape[0])
    else:
        grouper = group_by_columns
    for group_vars, df_g in df_lm_trajectory.groupby(grouper):
        df_lm_t_0 = msd.extract_first_timestep(df_g, time_col)
        df_g_msd_lm = df_g.groupby(time_col).apply(calculate_msd_lm, df_lm_t_0=df_lm_t_0)
        df_g_msd_lm[group_by_columns] = group_vars
        dfs.append(df_g_msd_lm)
    return pd.concat(dfs)


def calculate_msd_lm_by_dimension(
        df_lm_t: pd.DataFrame,
        df_lm_t_0: pd.DataFrame,
        dimensions: list[str]
) -> pd.Series:
    df_lm_t_vec = df_lm_t[dimensions].to_numpy()
    df_lm_t_0_vec = df_lm_t_0[dimensions].to_numpy()

    msd_lm_dims = (df_lm_t_vec - df_lm_t_0_vec) ** 2
    msd_lm_dims_avg = msd_lm_dims.mean(axis=0)
    msd_lm_dims_std = msd_lm_dims.std(axis=0) / np.sqrt(msd_lm_dims.shape[0])
    msd_lm_vec_avg = np.sum(msd_lm_dims, axis=1).mean()

    dr_dim_idx = ["dr_N_x^2", "dr_N_y^2", "dr_N_z^2"]
    dr_delta_dim_idx = [f"delta {s}" for s in dr_dim_idx]

    return pd.Series(
        [*msd_lm_dims_avg, *(msd_lm_dims_std * 3), msd_lm_vec_avg],
        index=[*dr_dim_idx, *dr_delta_dim_idx, "dR^2"]
    )


def calculate_msd_lm_by_dimension_df(
        df_lm_trajectory: pd.DataFrame,
        group_by_columns: list[str],
        time_col="t",
        dimensions: tuple[str, str, str] = ("R_x", "R_y", "R_z")
) -> pd.DataFrame:
    df_ete = df_lm_trajectory.reset_index(drop=False).drop("R", axis=1)
    dfs = []
    for params, df_group in df_ete.groupby(group_by_columns):
        df_group_t_0 = msd.extract_first_timestep(df_group, time_col)
        df_group_msd_lm = df_group.groupby(time_col).apply(
            calculate_msd_lm_by_dimension,
            df_lm_t_0=df_group_t_0,
            dimensions=list(dimensions)
        )
        df_group_msd_lm[group_by_columns] = params
        dfs.append(df_group_msd_lm)

    return pd.concat(dfs)
