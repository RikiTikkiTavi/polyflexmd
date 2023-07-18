import numpy as np
import pandas as pd


def calculate_msd(df_ete_t: pd.DataFrame, df_ete_t_0: pd.DataFrame) -> pd.Series:
    R_vec_cols = ["R_x", "R_y", "R_z"]

    df_ete_t_vec = df_ete_t[R_vec_cols].to_numpy()
    df_ete_t_0_vec = df_ete_t_0[R_vec_cols].to_numpy()

    msd = np.sum((df_ete_t_vec - df_ete_t_0_vec) ** 2, axis=1)

    msd_mean = msd.mean()
    msd_std_of_mean = msd.std() / msd.shape[0]

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
