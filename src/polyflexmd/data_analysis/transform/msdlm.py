import functools
import itertools
import logging
import multiprocessing
import typing

import numpy as np
import pandas as pd
import dask.dataframe
import polyflexmd.data_analysis.transform.constants as constants
import polyflexmd.data_analysis.transform.msd as msd
import polyflexmd.data_analysis.transform.transform as transform

_logger = logging.getLogger(__name__)

from scipy.optimize import curve_fit


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
        dimensions: tuple[str, str, str] = ("x", "y", "z")
) -> pd.DataFrame:
    df_lm_trajectory = df_lm_trajectory.reset_index(drop=False)
    dfs = []
    if len(group_by_columns) == 0:
        grouper = np.repeat(True, df_lm_trajectory.shape[0])
    else:
        grouper = group_by_columns
    for params, df_group in df_lm_trajectory.groupby(grouper):
        df_group_t_0 = msd.extract_first_timestep(df_group, time_col)
        df_group_msd_lm = df_group.groupby(time_col).apply(
            calculate_msd_lm_by_dimension,
            df_lm_t_0=df_group_t_0,
            dimensions=list(dimensions)
        )
        df_group_msd_lm[group_by_columns] = params
        dfs.append(df_group_msd_lm)

    return pd.concat(dfs)


def change_basis_df_lm_trajectory(df_lm_trajectory: pd.DataFrame, df_main_axis: pd.DataFrame):
    dims = ["x", "y", "z"]
    dfs = []
    for mol_id, df_mol in df_lm_trajectory.groupby("molecule-ID"):
        vec_axs = df_main_axis.loc[df_main_axis["molecule-ID"] == mol_id].iloc[0][dims].to_numpy()
        basis_new = transform.create_orthogonal_basis_with_given_vector(vec_axs)
        vecs = []
        for vec in df_mol[dims].to_numpy():
            vec_new = transform.basis_change_from_cartesian(basis_new, vec)
            vecs.append(vec_new)
        df_mol[dims] = vecs
        dfs.append(df_mol)

    return pd.concat(dfs)


def calculate_msd_alpha_df(df_msdlm: pd.DataFrame, n_bins: int, bins: typing.Optional[list[float]] = None, col: str = "dr_N^2"):

    def linregbin(df):
        if len(df) < 2:
            return pd.Series([np.NAN, np.NAN, np.NAN, np.NAN, np.NAN, np.NAN], index=["t/LJ", "alpha", "delta alpha", "delta t", "interval", "count"])
        f = lambda x, k: k * x
        xs = np.log10(df["t/LJ"])
        ys = np.log10(df[col])
        xs = xs - xs.min()
        ys = ys - ys.min()
        if f"delta {col}" in df.columns:
            dr = df[col]
            sigma_dr = df[f"delta {col}"] / 3
            dys = np.abs(1 / (dr * np.log(10)) * sigma_dr)
            popt, pcov = curve_fit(f, xs, ys, p0=(0.0), sigma=dys, absolute_sigma=True)
        else:
            popt, pcov = curve_fit(f, xs, ys, p0=(0.0))
        delta_alpha = np.sqrt(np.diag(pcov)[0]) * 3
        t_min = df["t/LJ"].min()
        t_max = df["t/LJ"].max()
        delta_t = (t_max - t_min) / 2
        t = t_min + delta_t
        return pd.Series(
            [t_min, popt[0], delta_alpha, delta_t, (t_min, t_max), df.shape[0]],
            index=["t/LJ", "alpha", "delta alpha", "delta t", "interval", "count"]
        )

    df_msdlm = df_msdlm.reset_index(drop=True)

    if bins is None:
        bins = np.logspace(
            np.log10(df_msdlm["t/LJ"].min()),
            np.log10(df_msdlm["t/LJ"].max()),
            n_bins,
            base=10
        ).tolist()
    binned_idx = pd.cut(df_msdlm["t/LJ"], bins=bins, precision=5, include_lowest=True)
    ks = df_msdlm.groupby(binned_idx).apply(linregbin)
    ks.set_index("t/LJ", inplace=True)
    return ks


def _calculate_msd_lm_df_proxy(df_lm_traj: pd.DataFrame, **kwargs):
    t = df_lm_traj.iloc[0]["t"]
    _logger.debug(f"Calculating MSDLM with t_start={t} ...")
    df_msdlm = calculate_msd_lm_df(df_lm_traj, **kwargs)
    df_msdlm = df_msdlm.reset_index()
    df_msdlm["t"] = df_msdlm["t"] - df_msdlm["t"].min()
    return df_msdlm


def calculate_msdlm_mean_avg_over_t_start(
        df_lm_traj: pd.DataFrame,
        group_by_columns: list[str],
        n_workers: int,
        t_start: int,
        exclude_n_last: int = 10,
        take_n_first: typing.Optional[int] = None,
        chunk_size: int = 2,
) -> pd.DataFrame:
    df_lm_traj = df_lm_traj[df_lm_traj["t"] % 10000 == 0]
    t_0 = df_lm_traj["t"].min()
    t_start = t_0 + t_start
    ts = sorted(df_lm_traj.loc[df_lm_traj["t"] >= t_start]["t"].unique())[:-exclude_n_last]

    if take_n_first is not None:
        ts = ts[:take_n_first]

    with multiprocessing.Pool(processes=n_workers) as pool:
        dfs = pool.imap(
            functools.partial(
                _calculate_msd_lm_df_proxy,
                group_by_columns=group_by_columns
            ),
            (df_lm_traj.loc[df_lm_traj["t"] >= t] for t in ts),
            chunksize=chunk_size
        )

        _logger.debug("Concatenating and calculating mean ...")
        return pd.concat(dfs).groupby(["t", *group_by_columns]).mean()
