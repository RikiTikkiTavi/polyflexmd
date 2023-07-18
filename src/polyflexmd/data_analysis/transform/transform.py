import logging
import multiprocessing
import time
import typing

import dask.dataframe
import numpy as np
import pandas as pd
import enum
import polyflexmd.data_analysis.data.types as types
import pathlib
import functools

import scipy.optimize

import polyflexmd.data_analysis.theory.kremer_grest

_logger = logging.getLogger(__name__)


class AtomGroup(enum.Enum):
    ROOT = 1
    FREE = 2
    LEAF = 3


def unfold_coordinate(val: float, i: float, box_length: float):
    return val + i * box_length


def unfold_coordinates_row(traj_row: pd.Series, system_data: types.LammpsSystemData) -> pd.Series:
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
        system_data: types.LammpsSystemData
) -> pd.DataFrame:
    _logger.debug("Unfolding coordinates ...")

    dimensions = ('x', 'y', 'z')

    for dim_i, dim_name in enumerate(dimensions):
        box_length = system_data.box.bounds[dim_i][1] - system_data.box.bounds[dim_i][0]
        trajectory_df[dim_name] = trajectory_df[dim_name] + trajectory_df[f"i{dim_name}"] * box_length

    return trajectory_df


def calculate_end_to_end(molecule_traj_step_df_unf: pd.DataFrame) -> pd.Series:

    root_atom_data: pd.Series = molecule_traj_step_df_unf \
        .loc[molecule_traj_step_df_unf["type"] == AtomGroup.ROOT.value] \
        .head(5) \
        .sort_values("id", ascending=True) \
        .iloc[0]

    leaf_atom_data: pd.Series = molecule_traj_step_df_unf \
        .loc[molecule_traj_step_df_unf["type"] == AtomGroup.LEAF.value] \
        .head(5) \
        .sort_values("id", ascending=False) \
        .iloc[0]

    dimensions = ['x', 'y', 'z']

    r_root = root_atom_data[dimensions].to_numpy()
    r_leaf = leaf_atom_data[dimensions].to_numpy()

    R_vec = r_leaf - r_root
    R = np.linalg.norm(R_vec)

    return pd.Series(data=[*R_vec, R], index=["R_x", "R_y", "R_z", "R"])


def join_raw_trajectory_df_with_system_data(
        raw_trajectory_df: dask.dataframe.DataFrame,
        system_data: types.LammpsSystemData
) -> pd.DataFrame:
    atom_id_to_molecule_id = {row[0]: row[1]["molecule-ID"] for row in system_data.atoms.iterrows()}
    _logger.debug("Joining with system data ...")
    raw_trajectory_df["molecule-ID"] = raw_trajectory_df["id"].map(atom_id_to_molecule_id).astype(np.ushort)
    return raw_trajectory_df


def calc_end_to_end_df(
        trajectory_df_unfolded: pd.DataFrame,
        group_by_params: typing.Iterable[str] = tuple()
) -> dask.dataframe.DataFrame:
    gb = trajectory_df_unfolded.groupby(["t", *group_by_params, "molecule-ID"])
    return gb.apply(calculate_end_to_end, meta=pd.DataFrame(columns=["R_x", "R_y", "R_z", "R"]))


def calculate_ete_change_ens_avg(df_ete_t: pd.DataFrame, df_ete_t_0: pd.DataFrame) -> float:
    R_vec_cols = ["R_x", "R_y", "R_z"]

    df_ete_t_vec = df_ete_t[R_vec_cols].to_numpy()
    df_ete_t_0_vec = df_ete_t_0[R_vec_cols].to_numpy()

    return np.sum((df_ete_t_vec - df_ete_t_0_vec) ** 2, axis=1).mean()


def calculate_ete_change_ens_avg_df(df_ete: pd.DataFrame) -> pd.DataFrame:
    t_min = df_ete.index.get_level_values("t").min()
    ete_df_t_0 = df_ete.loc[pd.IndexSlice[:, t_min], :]

    return df_ete \
        .groupby(level="t") \
        .apply(functools.partial(calculate_ete_change_ens_avg, df_ete_t_0=ete_df_t_0))


def calculate_neigh_distance_avg(mol_traj_step_df_unf: pd.DataFrame) -> float:
    dims = ['x', 'y', 'z']
    mol_traj_step = mol_traj_step_df_unf[dims].to_numpy()
    return np.sum((mol_traj_step[1:] - mol_traj_step[:-1]) ** 2, axis=1).mean()


def calculate_neigh_distance_avg_df(
        trajectory_df_unfolded: pd.DataFrame,
        t_equilibrium: float
) -> float:
    trajectory_df_unfolded_equi = trajectory_df_unfolded.loc[trajectory_df_unfolded["t"] > t_equilibrium]

    l_avg_chains = []

    for molecule_id, df_mol in trajectory_df_unfolded_equi.groupby(["t", "molecule-ID"]):
        l_avg_chains.append(calculate_neigh_distance_avg(df_mol))

    return np.mean(l_avg_chains)


def calculate_bond_lengths(mol_traj_step_df_unf: pd.DataFrame) -> np.ndarray:
    dims = ['x', 'y', 'z']
    mol_traj_step = mol_traj_step_df_unf[dims].to_numpy()
    return np.sqrt(np.sum((mol_traj_step[1:] - mol_traj_step[:-1]) ** 2, axis=1))


def extract_bond_lengths_df(
        trajectory_df_unfolded: pd.DataFrame,
        t_equilibrium: float,
) -> list[float]:
    trajectory_df_unfolded_equi = trajectory_df_unfolded.loc[trajectory_df_unfolded["t"] > t_equilibrium]

    bond_lengths = []

    for molecule_id, df_mol in trajectory_df_unfolded_equi.groupby(["t", "molecule-ID"]):
        bond_lengths.extend(calculate_bond_lengths(df_mol))

    return bond_lengths


def extract_bond_lengths_df_kappas(
        trajectory_df_unfolded_kappas: pd.DataFrame,
        t_equilibrium: float
):
    return trajectory_df_unfolded_kappas \
        .groupby("kappa") \
        .parallel_apply(extract_bond_lengths_df, t_equilibrium=t_equilibrium) \
        .apply(pd.Series) \
        .reset_index() \
        .melt(
        id_vars=["kappa"],
        var_name="i",
        value_name="l_b"
    ).set_index(["kappa", "i"])


def calculate_ete_sq_t_avg_df(df_ete_mean: pd.DataFrame, t_equilibrium: float) -> float:
    df_ete_mean = df_ete_mean.reset_index()
    df_ete_equi = df_ete_mean.loc[df_ete_mean["t"] > t_equilibrium]
    return df_ete_equi["R^2"].mean()


def calculate_ete_sq_t_avg_df_kappas(df_ete_mean_kappas: pd.DataFrame, t_equilibrium: float) -> pd.DataFrame:
    return pd.DataFrame(
        df_ete_mean_kappas.groupby("kappa").apply(calculate_ete_sq_t_avg_df, t_equilibrium=t_equilibrium),
        columns=["R^2"])


def calculate_ete_sq_t_avg_df_kappas_dend(df_ete_mean_kappas: pd.DataFrame, t_equilibrium: float) -> pd.DataFrame:
    return pd.DataFrame(
        df_ete_mean_kappas.groupby(["kappa", "d_end"]).apply(calculate_ete_sq_t_avg_df, t_equilibrium=t_equilibrium),
        columns=["R^2"])


def calculate_contour_length(mol_traj_step_df_unf: pd.DataFrame) -> float:
    dims = ['x', 'y', 'z']
    mol_traj_step = mol_traj_step_df_unf[dims].to_numpy()
    # noinspection PyTypeChecker
    return np.sum(np.sqrt(np.sum((mol_traj_step[1:] - mol_traj_step[:-1]) ** 2, axis=1)))


def calculate_contour_length_df(trajectory_df_unfolded: pd.DataFrame) -> pd.DataFrame:
    return trajectory_df_unfolded.groupby(["molecule-ID", "t"]).apply(calculate_contour_length)


def calculate_ens_avg_df_ete_change_kappas(df_ete_kappas: pd.DataFrame) -> pd.DataFrame:
    dfs_ete_change_kappas = []
    for kappa, df_ete_kappa in df_ete_kappas.groupby("kappa"):
        df_ete_change_kappa = pd.DataFrame(calculate_ete_change_ens_avg_df(df_ete_kappa.droplevel("kappa")),
                                           columns=["dR^2"])
        df_ete_change_kappa["kappa"] = kappa
        dfs_ete_change_kappas.append(df_ete_change_kappa)
    return pd.concat(dfs_ete_change_kappas)


def calculate_ens_avg_df_ete_change_kappas_dend(df_ete_kappas_dend: pd.DataFrame) -> pd.DataFrame:
    dfs_ete_change = []
    for (kappa, d_end), df_ete in df_ete_kappas_dend.groupby(["kappa", "d_end"]):
        df_ete_change = pd.DataFrame(calculate_ete_change_ens_avg_df(df_ete.droplevel(["kappa", "d_end"])),
                                     columns=["dR^2"])
        df_ete_change["kappa"] = kappa
        df_ete_change["d_end"] = d_end
        dfs_ete_change.append(df_ete_change)
    return pd.concat(dfs_ete_change)


def bond_auto_correlation(idx: np.ndarray, l_p: float, l_b: float) -> np.ndarray:
    return np.exp(-np.abs(idx[0] - idx[1]) * l_b / l_p)


def extract_cos_matrix_from_chain(positions: np.ndarray) -> np.array:
    bonds_molecule = positions[1:] - positions[:-1]
    bonds_molecule /= np.linalg.norm(bonds_molecule, axis=1)[:, np.newaxis]
    return bonds_molecule @ bonds_molecule.T


def estimate_kuhn_length(
        traj_df_unf: pd.DataFrame,
        N_beads: int,
        l_K_guess: typing.Optional[float] = None,
        l_b: typing.Optional[float] = None
) -> pd.Series:
    traj_df_unf.sort_values(by="id", ascending=True, inplace=True)

    if l_b is not None and l_K_guess is None and "kappa" in traj_df_unf.columns:
        kappa = float(traj_df_unf.iloc[0]["kappa"])
        l_K_guess = np.float32(polyflexmd.data_analysis.theory.kremer_grest.bare_kuhn_length(
            kappa=kappa,
            l_b=l_b
        ))

    cos_matrices_molecules = [
        extract_cos_matrix_from_chain(df[['x', 'y', 'z']].to_numpy())
        for _, df in traj_df_unf.groupby("molecule-ID")
    ]

    cos_matrix_avg = np.mean(cos_matrices_molecules, axis=0)
    # angle_matrix_std = np.std(angle_matrices_molecules, axis=0)

    indexes_up = np.triu_indices(N_beads - 1, k=1)
    indexes_down = np.tril_indices(N_beads - 1, k=-1)
    x_data = np.hstack([indexes_up, indexes_down])
    row_idx, col_idx = x_data
    y_data = cos_matrix_avg[row_idx, col_idx]

    l_p_guess = l_K_guess / 2

    popt, pcov = scipy.optimize.curve_fit(
        functools.partial(bond_auto_correlation, l_b=l_b),
        x_data,
        y_data,
        p0=l_p_guess,
        # sigma=y_std,
        # absolute_sigma=True,
        bounds=[l_p_guess / 100, l_p_guess * 5],
    )

    l_p = popt[0]
    dl_p = np.sqrt(np.diag(pcov))[0]

    # dl_p corresponds to 1 sigma

    l_K = l_p * 2
    d_l_K = dl_p * 2 * 3

    _logger.debug(f"Estimated l_K={l_K} +- {d_l_K} vs Guess l_K={l_K_guess}")

    return pd.Series({"l_K": l_K, "d_l_K": d_l_K})


def estimate_kuhn_length_df(
        df_trajectory: dask.dataframe.DataFrame,
        group_by_params: list[str],
        N_beads: int,
        l_b: typing.Optional[float] = None,
        time_col: str = "t"
) -> dask.dataframe.DataFrame:
    l_K_results = df_trajectory.groupby([time_col, *group_by_params]).apply(
        estimate_kuhn_length,
        l_b=l_b,
        N_beads=N_beads,
        meta=pd.DataFrame(columns=["l_K", "d_l_K"])
    )

    return l_K_results


def time_LJ_to_REAL(t_LJ, L_contour):
    import scipy.constants
    T_GRILL = 23 + 273
    eps = scipy.constants.k / T_GRILL

    M_r_EEA1 = 162  # kg/mol
    N_beads = 64
    m_bead = 1
    m_EEA1 = M_r_EEA1 / scipy.constants.N_A
    m = m_EEA1 / (N_beads * m_bead)

    L_GRILL = 230 * 10e-9  # m
    sigma = L_GRILL / L_contour

    return t_LJ / np.sqrt(eps / (m * sigma ** 2))


def create_orthogonal_basis_with_given_vector(v: np.ndarray) -> np.ndarray:
    assert v.shape == (3,)

    v = v / np.linalg.norm(v)

    e_x = np.array([1.0, 0.0, 0.0])
    e_y = np.array([0.0, 1.0, 0.0])

    if not np.allclose(e_x, v):
        b = np.cross(e_x, v)
    else:
        b = np.cross(e_x + e_y, v)

    b = b / np.linalg.norm(b)

    c = np.cross(v, b)
    c = c / np.linalg.norm(c)

    assert np.allclose(c.dot(v), 0)
    assert np.allclose(c.dot(b), 0)
    assert np.allclose(b.dot(v), 0)

    basis = np.array([c, b, v])

    assert np.allclose(basis @ basis.T, np.identity(3))

    return np.array([c, b, v])


def basis_change_from_cartesian(basis_new: np.ndarray, v_old: np.ndarray):
    return np.linalg.solve(basis_new.T, v_old)


def change_basis_df_ete(df_ete: pd.DataFrame, df_main_axis: pd.DataFrame):
    dims = ["x", "y", "z"]
    dims_R = ["R_x", "R_y", "R_z"]
    dfs = []
    for mol_id, df_mol in df_ete.groupby("molecule-ID"):
        vec_axs = df_main_axis.loc[df_main_axis["molecule-ID"] == mol_id].iloc[0][dims].to_numpy()
        basis_new = create_orthogonal_basis_with_given_vector(vec_axs)
        R_vecs = []
        for R_vec in df_mol[dims_R].to_numpy():
            R_vec_new = basis_change_from_cartesian(basis_new, R_vec)
            R_vecs.append(R_vec_new)
        df_mol[dims_R] = R_vecs
        dfs.append(df_mol)

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
    MSD_vec_avg = np.sum(MSD_dims, axis=1).mean()
    return pd.Series([*MSD_dims_avg, MSD_vec_avg], index=["dR_x^2", "dR_y^2", "dR_z^2", "dR^2"])


def calculate_msd_by_dimension_df(
        df_ete: pd.DataFrame,
        group_by_params: list[str],
        time_param: str = "t",
        dimensions: tuple[str, str, str] = ("R_x", "R_y", "R_z")
) -> pd.DataFrame:
    df_ete = df_ete.reset_index(drop=False).drop("R", axis=1)
    dfs = []
    for params, df_group in df_ete.groupby(group_by_params):
        t_min = df_group[time_param].min()
        df_group_t_0 = df_group.loc[df_group["t"] == t_min]
        df_group_MSD = df_group.groupby(time_param).apply(
            calculate_msd_by_dimension,
            df_ete_step_0=df_group_t_0,
            dimensions=list(dimensions)
        )
        df_group_MSD[group_by_params] = params
        dfs.append(df_group_MSD)

    return pd.concat(dfs)