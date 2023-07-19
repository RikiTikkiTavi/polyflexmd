import numpy as np
import pandas as pd


def calculate_msd(df_ete_t: pd.DataFrame, df_ete_t_0: pd.DataFrame) -> pd.Series:
    R_vec_cols = ["R_x", "R_y", "R_z"]

    df_ete_t_vec = df_ete_t[R_vec_cols].to_numpy()
    df_ete_t_0_vec = df_ete_t_0[R_vec_cols].to_numpy()

    msd = np.sum((df_ete_t_vec - df_ete_t_0_vec) ** 2, axis=1)

    msd_mean = msd.mean()
    msd_std_of_mean = msd.std() / np.sqrt(msd.shape[0])

    return pd.Series([msd_mean, 3 * msd_std_of_mean], index=["dR^2", "delta dR^2"])


def extract_first_timestep(df: pd.DataFrame, time_col="t") -> pd.DataFrame:
    t_min = df[time_col].min()
    return df.loc[df[time_col] == t_min]


def calculate_msd_df(df_ete: pd.DataFrame, group_by_columns: list[str], time_col="t"):
    df_ete = df_ete.reset_index()
    dfs = []
    for group_vars, df_g in df_ete.groupby(group_by_columns):
        df_ete_t_0 = extract_first_timestep(df_g, time_col)
        df_g_msd = df_g.groupby(time_col).apply(calculate_msd, df_ete_t_0=df_ete_t_0)
        df_g_msd[group_by_columns] = group_vars
        dfs.append(df_g_msd)
    return pd.concat(dfs)


def calculate_msd_by_dimension(
        df_ete_step: pd.DataFrame,
        df_ete_step_0: pd.DataFrame,
        dimensions: list[str]
) -> pd.Series:
    df_ete_step_vec = df_ete_step[dimensions].to_numpy()
    df_ete_step_0_vec = df_ete_step_0[dimensions].to_numpy()
    MSD_dims = (df_ete_step_vec - df_ete_step_0_vec) ** 2
    MSD_dims_avg = MSD_dims.mean(axis=0)
    MSD_dims_std = MSD_dims.std(axis=0) / np.sqrt(MSD_dims.shape[0])
    MSD_vec_avg = np.sum(MSD_dims, axis=1).mean()

    dr_dim_idx = ["dR_x^2", "dR_y^2", "dR_z^2"]
    dr_delta_dim_idx = [f"delta {s}" for s in dr_dim_idx]
    return pd.Series(
        [*MSD_dims_avg, *(MSD_dims_std * 3), MSD_vec_avg],
        index=["dR_x^2", "dR_y^2", "dR_z^2", *dr_delta_dim_idx, "dR^2"]
    )


def calculate_msd_by_dimension_df(
        df_ete: pd.DataFrame,
        group_by_params: list[str],
        time_param: str = "t",
        dimensions: tuple[str, str, str] = ("R_x", "R_y", "R_z")
) -> pd.DataFrame:
    df_ete = df_ete.reset_index(drop=False).drop("R", axis=1)
    dfs = []
    for params, df_group in df_ete.groupby(group_by_params):
        df_group_t_0 = extract_first_timestep(df_group, time_param)
        df_group_MSD = df_group.groupby(time_param).apply(
            calculate_msd_by_dimension,
            df_ete_step_0=df_group_t_0,
            dimensions=list(dimensions)
        )
        df_group_MSD[group_by_params] = params
        dfs.append(df_group_MSD)

    return pd.concat(dfs)
