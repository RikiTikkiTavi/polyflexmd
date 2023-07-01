import multiprocessing
import time
import typing

import numpy as np
import pandas as pd
import enum
import polyflexmd.data_analysis.data.types as types
import pathlib
import functools

import scipy.optimize

import polyflexmd.data_analysis.theory.kremer_grest


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
    trajectory_df_unfolded = trajectory_df.copy()
    dimensions = ('x', 'y', 'z')

    for dim_i, dim_name in enumerate(dimensions):
        box_length = system_data.box.bounds[dim_i][1] - system_data.box.bounds[dim_i][0]
        trajectory_df_unfolded[dim_name] = trajectory_df[dim_name] + trajectory_df[f"i{dim_name}"] * box_length

    return trajectory_df_unfolded


def calculate_end_to_end(molecule_traj_step_df_unf: pd.DataFrame) -> pd.Series:
    root_atom_data: pd.Series = molecule_traj_step_df_unf \
        .loc[molecule_traj_step_df_unf["type"] == AtomGroup.ROOT.value] \
        .sort_values("id", ascending=True) \
        .iloc[0]

    leaf_atom_data: pd.Series = molecule_traj_step_df_unf \
        .loc[molecule_traj_step_df_unf["type"] == AtomGroup.LEAF.value] \
        .sort_values("id", ascending=False) \
        .iloc[0]

    dimensions = ['x', 'y', 'z']

    r_root = root_atom_data[dimensions].to_numpy()
    r_leaf = leaf_atom_data[dimensions].to_numpy()

    R_vec = r_leaf - r_root
    R = np.linalg.norm(R_vec)

    return pd.Series(data=[*R_vec, R], index=["R_x", "R_y", "R_z", "R"])


def join_raw_trajectory_df_with_system_data(
        raw_trajectory_df: pd.DataFrame,
        system_data: types.LammpsSystemData
) -> pd.DataFrame:
    return raw_trajectory_df.join(
        system_data.atoms["molecule-ID"],
        on="id"
    )


def calc_end_to_end_df(
        trajectory_df_unfolded: pd.DataFrame,
        group_by_params: typing.Iterable[str] = tuple(),
        parallel: bool = False
) -> pd.DataFrame:
    gb = trajectory_df_unfolded.groupby([*group_by_params, "molecule-ID"])
    if parallel:
        return gb.parallel_apply(lambda dfg: dfg.groupby(["t"]).apply(calculate_end_to_end))
    else:
        return gb.apply(lambda dfg: dfg.groupby(["t"]).apply(calculate_end_to_end))


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


def bond_auto_correlation(idx: np.ndarray, l_p: float) -> np.ndarray:
    return np.exp(-np.abs(idx[0] - idx[1]) / l_p)


def _extract_angles_from_chain(df: pd.DataFrame) -> np.array:
    dims = ['x', 'y', 'z']
    mol_traj_step: np.ndarray = df[dims].to_numpy()
    bonds_molecule = mol_traj_step[1:] - mol_traj_step[:-1]
    bonds_molecule_normalized = bonds_molecule / np.sqrt(np.sum(bonds_molecule ** 2, axis=1))[:, np.newaxis]
    return bonds_molecule_normalized @ bonds_molecule_normalized.T


def estimate_kuhn_length(
        traj_df_unf: pd.DataFrame,
        l_K_guess: typing.Optional[float] = None,
        l_b: typing.Optional[float] = None,
        n_processes: int = 16,
        time_col: str = "t"
) -> pd.Series:

    if l_b is not None and l_K_guess is None and "kappa" in traj_df_unf.columns:
        l_K_guess = polyflexmd.data_analysis.theory.kremer_grest.bare_kuhn_length(
            kappa=float(traj_df_unf.iloc[0]["kappa"]),
            l_b=l_b
        )

    p: multiprocessing.Pool
    with multiprocessing.Pool(processes=n_processes) as p:
        angle_matrices_molecules = p.map(
            _extract_angles_from_chain,
            [df for _, df in traj_df_unf.groupby([time_col, "molecule-ID"])]
        )

    angle_matrix_avg = np.mean(angle_matrices_molecules, axis=0)

    x_data = np.array(list(np.ndindex(*angle_matrix_avg.shape))).T
    y_data = np.array([angle_matrix_avg[idx] for idx in np.ndindex(*angle_matrix_avg.shape)])

    l_p_guess = l_K_guess / 2

    popt, pcov = scipy.optimize.curve_fit(
        bond_auto_correlation,
        x_data,
        y_data,
        p0=l_p_guess,
        bounds=[l_p_guess / 5, l_p_guess * 5],
    )

    l_p = popt[0]
    dl_p = np.sqrt(np.diag(pcov))[0]

    # dl_p corresponds to 1 sigma

    return pd.Series({"l_K": l_p * 2, "d_l_K": dl_p * 2 * 3})


def estimate_kuhn_length_df(
        df_trajectory: pd.DataFrame,
        group_by_params: list[str],
        l_b: typing.Optional[float] = None,
        n_processes: int = 16,
        time_col: str = "t",
        t_equilibrium: float = 0.0
) -> typing.Union[pd.DataFrame, pd.Series]:
    if t_equilibrium > 0.0:
        df_trajectory = df_trajectory.loc[df_trajectory[time_col] > t_equilibrium]

    l_K_result = df_trajectory.groupby(group_by_params).apply(
        estimate_kuhn_length,
        time_col=time_col,
        l_b=l_b,
        n_processes=n_processes
    )

    return l_K_result


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


def main():
    vec = np.array([-0.15992717, -0.01745058, -0.95656614])
    basis_new = create_orthogonal_basis_with_given_vector(vec)
    R_vec_old = np.array([21.1113, 3.337980, -51.8683])
    R_vec_new = basis_change_from_cartesian(basis_new, R_vec_old)
    print(f"R_vec old: {R_vec_old}; Norm: {np.linalg.norm(R_vec_old)}")
    print(f"R_vec new: {R_vec_new}; Norm: {np.linalg.norm(R_vec_new)}")
