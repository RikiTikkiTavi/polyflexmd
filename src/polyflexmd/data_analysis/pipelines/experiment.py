import copy
import logging
import pathlib
import time
import typing

import numpy as np
import pandas as pd
from pandarallel import pandarallel

import polyflexmd.experiment_runner.config
import polyflexmd.data_analysis.pipelines.trajectory
import polyflexmd.data_analysis.data.read
import polyflexmd.data_analysis.data.constants
import polyflexmd.data_analysis.transform.transform as transform

import polyflexmd.data_analysis.theory.kremer_grest

import dask.dataframe

_logger = logging.getLogger(__name__)


def process_experiment_data(
        path_experiment: pathlib.Path,
        style: typing.Literal["l_K+d_end", "l_K", "simple"],
        n_workers: int = 16,
        read_relax: bool = False,
        enable_l_K_estimate: bool = True,
        time_steps_per_partition: int = 100000,
        save_angle_matrix: bool = False
):
    _logger.info(f"Processing data of experiment: {path_experiment}")
    _logger.info(f"Data style: {style}")
    _logger.info(f"Read relax: {read_relax}")
    _logger.info(f"time_steps_per_partition: {time_steps_per_partition}")

    pandarallel.initialize(
        nb_workers=n_workers,
        progress_bar=False,
        use_memory_fs=None
    )

    path_data_raw = path_experiment / "data" / "raw"

    # Handle legacy where raw does not exist
    if not path_data_raw.exists():
        path_data_raw = path_experiment / "data"

    _logger.debug(f"Raw data path: {path_data_raw}")

    path_config = next(path_experiment.glob("*.toml"))
    _logger.debug(f"Config path: {path_config}")

    path_initial_system = path_experiment / "data" / "initial_system.data"

    _logger.debug(f"Initial system path: {path_initial_system}")

    config = polyflexmd.experiment_runner.config.read_experiment_config(path_config)

    kappas: typing.Optional[list[float]] = None
    d_ends: typing.Optional[list[float]] = None

    group_by_parameters = []

    if style.startswith("l_K"):
        kappas = [
            config.simulation_config.variables["kappa_start"] + config.simulation_config.variables["kappa_delta"] * i
            for i in range(config.simulation_config.variables["kappa_n_values"])
        ]
        group_by_parameters.append("kappa")
    if style == "l_K+d_end":
        d_ends = [
            config.simulation_config.variables["d_end_start"] + config.simulation_config.variables["d_end_delta"] * i
            for i in range(config.simulation_config.variables["d_end_n_values"])
        ]
        group_by_parameters.append("d_end")

    _logger.debug(f"kappas: {kappas}")
    _logger.debug(f"d_ends: {d_ends}")

    initial_system = polyflexmd.data_analysis.data.read.read_lammps_system_data(
        path=path_initial_system
    )

    path_data_processed = path_experiment / "data" / "processed"
    path_data_processed.mkdir(exist_ok=True, parents=True)

    # Main axis
    path_df_main_axis = path_data_processed / "main_axis.csv"
    if path_df_main_axis.exists():
        _logger.info(f"{path_df_main_axis} exists;")
    else:
        _logger.info(f"{path_df_main_axis} does not exist;")
        _logger.info(f"Extracting main axis ...")
        df_main_axis = transform.unfold_coordinates_df(
            initial_system.atoms, initial_system
        ).groupby("molecule-ID").head(n=2)
        _logger.info(f"Writing {path_df_main_axis} ...")
        df_main_axis.to_csv(path_df_main_axis, index=False)

    # Trajectories
    path_traj_dir = path_data_processed / "trajectories"
    traj_glob = f"{str(path_traj_dir)}/trajectories-*.csv"

    path_traj_processed = path_data_processed / "trajectories.csv"

    if path_traj_dir.exists() or path_traj_processed.exists():
        if path_traj_processed.exists():
            traj_glob = path_traj_processed

        _logger.info(f"{path_traj_dir} exists => Reading processed ...")
        traj_column_types = copy.deepcopy(polyflexmd.data_analysis.data.constants.RAW_TRAJECTORY_DF_COLUMN_TYPES)
        traj_column_types.pop("ix")
        traj_column_types.pop("iy")
        traj_column_types.pop("iz")
        for param in group_by_parameters:
            traj_column_types[param] = "category"
        df_trajectories = dask.dataframe.read_csv(
            traj_glob,
            dtype=traj_column_types
        )
        divisions = df_trajectories["t"].loc[
            df_trajectories["t"] % time_steps_per_partition == 0
            ].unique().compute().sort_values().tolist()
        df_trajectories = df_trajectories.set_index("t", divisions=divisions)
        df_trajectories.persist()

    else:
        _logger.info(f"{path_traj_dir} does not exist;")
        _logger.info("Reading and processing trajectories ...")

        total_time_steps = 0
        if "n_equilibrium_steps" in config.simulation_config.variables:
            total_time_steps += int(config.simulation_config.variables["n_equilibrium_steps"])
        elif "n_equilibrium_steps_1" in config.simulation_config.variables and "n_equilibrium_steps_2" in config.simulation_config.variables:
            total_time_steps += config.simulation_config.variables["n_equilibrium_steps_1"]
            total_time_steps += config.simulation_config.variables["n_equilibrium_steps_2"]
        if read_relax:
            total_time_steps += config.simulation_config.variables["n_relax_steps"]

        _logger.debug(f"Total time steps: {total_time_steps}")

        df_trajectories = polyflexmd.data_analysis.pipelines.trajectory.read_and_process_trajectories(
            trajectories=polyflexmd.data_analysis.data.read.get_experiment_trajectories_paths(
                experiment_raw_data_path=path_data_raw,
                style=style,
                kappas=kappas,
                d_ends=d_ends,
                read_relax=read_relax
            ),
            system=initial_system,
            time_steps_per_partition=time_steps_per_partition,
            total_time_steps=total_time_steps if total_time_steps > 0 else None
        )
        df_trajectories.persist()
        path_traj_dir.mkdir(exist_ok=True, parents=True)
        _logger.info(f"Writing {traj_glob} ...")
        df_trajectories.to_csv(traj_glob, single_file=False, index=True)

    # df_trajectories = df_trajectories.compute()

    # ETE
    path_ete = path_data_processed / "ete.csv"

    if path_ete.exists():
        _logger.info(f"{path_ete} exists => Reading processed ...")
        df_ete = pd.read_csv(path_ete, index_col=[*group_by_parameters, "molecule-ID", "t"])
    else:
        _logger.info(f"{path_ete} does not exist;")
        _logger.info(f"Calculating ETE ...")
        _logger.debug(f"Group by params for ete calculation: {group_by_parameters}")
        df_ete = transform.calc_end_to_end_df(
            df_trajectories,
            group_by_params=group_by_parameters,
        ).compute()
        _logger.info(f"Writing {path_ete} ...")
        df_ete.to_csv(path_ete, index=True)

    # l_K estimate
    if enable_l_K_estimate:

        path_df_l_K = path_data_processed / "l_K-estimate.csv"

        if path_df_l_K.exists():
            _logger.info(f"{path_df_l_K} exists;")
        else:
            _logger.info(f"{path_df_l_K} does not exist;")
            _logger.info(f"Estimating l_K from trajectories ...")
            t_start = time.time()
            l_ks_estimate = transform.estimate_kuhn_length_df(
                df_trajectory=df_trajectories,
                group_by_params=group_by_parameters,
                l_b=config.initial_system_config.system_config.bond_length,
                N_beads=config.initial_system_config.system_config.n_monomers
            ).compute()
            _logger.debug(f"Estimatation of l_K took: {time.time() - t_start}s")
            _logger.info(f"Writing {path_df_l_K}")
            l_ks_estimate.to_csv(path_df_l_K, index=True)

    # MSD
    path_msd = path_data_processed / "msd.csv"

    if path_msd.exists():
        _logger.info(f"{path_msd} exists;")
    else:
        _logger.info(f"{path_msd} does not exist;")
        _logger.info(f"Calculating MSD ...")

        df_ete_equi = df_ete.iloc[
            df_ete.index.get_level_values("t") >= config.simulation_config.variables["n_relax_steps"]]

        df_ete_equi = df_ete_equi.reset_index().set_index([*group_by_parameters, "molecule-ID", "t"])

        if style == "l_K":
            df_msd = transform.calculate_ens_avg_df_ete_change_kappas(df_ete_equi)
        elif style == "l_K+d_end":
            df_msd = transform.calculate_ens_avg_df_ete_change_kappas_dend(df_ete_equi)
        else:
            df_msd = transform.calculate_ete_change_ens_avg_df(df_ete_equi)

        df_msd["t/LJ"] = df_msd.index.get_level_values("t").map(lambda t: t * 0.0025)
        df_msd["t/LJ"] = df_msd["t/LJ"] - df_msd["t/LJ"].min()

        _logger.info(f"Writing {path_msd} ...")
        df_msd.to_csv(path_msd, index=True)
