import copy
import platform

import dask.dataframe
import numpy as np
import pathlib
import typing
from enum import Enum
import logging

import pandas as pd

import polyflexmd.data_analysis.transform.msdlm
import polyflexmd.data_analysis.data.constants
import typer

from dask_jobqueue import SLURMCluster
import dask.distributed

import polyflexmd.data_analysis.utils

app = typer.Typer(invoke_without_command=False)

_logger = logging.getLogger(__name__)


class DataStyle(str, Enum):
    l_K = "l_K"
    l_K_d_end = "l_K+d_end"
    simple = "simple"


@app.command("calculate-msdlm-avg-over-t-start")
def calculate_msdlm_avg_over_t_start(
        path_df_lm_traj: pathlib.Path,
        output_path: pathlib.Path,
        style: typing.Annotated[
            DataStyle,
            typer.Option(case_sensitive=False)
        ],
        n_workers: typing.Annotated[
            int,
            typer.Option()
        ] = 16,
        t_start: typing.Annotated[
            int,
            typer.Option()
        ] = 400,
        exclude_n_last: typing.Annotated[
            int,
            typer.Option()
        ] = 10,
        take_n_first: typing.Annotated[
            typing.Optional[int],
            typer.Option()
        ] = None,
        chunk_size: typing.Annotated[
            int,
            typer.Option()
        ] = 2
):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s :: %(message)s', datefmt='%d.%m.%Y %I:%M:%S'
    )

    variables = []
    if style == DataStyle.l_K:
        variables.append("kappa")
    elif style == DataStyle.l_K_d_end:
        variables.extend(["kappa", "d_end"])

    if output_path.exists():
        raise ValueError("Output file already exists.")

    _logger.info(f"Calculating MSDLM avg over start time using n_workers={n_workers} ...")

    dtypes = {
        "t": np.uint64,
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
        "molecule-ID": np.uint16
    }
    for var in variables:
        dtypes[var] = "category"

    df_lm_traj = pd.read_csv(path_df_lm_traj, dtype=dtypes)

    df_msd = polyflexmd.data_analysis.transform.msdlm.calculate_msdlm_mean_avg_over_t_start(
        df_lm_traj=df_lm_traj,
        n_workers=n_workers,
        t_start=t_start,
        exclude_n_last=exclude_n_last,
        group_by_columns=variables,
        take_n_first=take_n_first,
        chunk_size=chunk_size
    )

    _logger.debug(f"Writing to {output_path}...")
    df_msd.to_csv(output_path, index=True)


@app.command("extract-fm-trajectory")
def extract_first_monomer_trajectory(
        path_trajectories: str,
        style: typing.Annotated[
            DataStyle,
            typer.Option(case_sensitive=False)
        ],
        output_path: typing.Annotated[
            pathlib.Path,
            typer.Option()
        ],
        partition: typing.Annotated[
            str,
            typer.Option()
        ] = "haswell",
        reservation: typing.Annotated[
            typing.Optional[str],
            typer.Option()
        ] = None,
        cores: typing.Annotated[
            int,
            typer.Option()
        ] = 12,
        memory: typing.Annotated[
            str,
            typer.Option()
        ] = "60GB",
        account: typing.Annotated[
            str,
            typer.Option()
        ] = "p_mdpolymer",
        max_workers: typing.Annotated[
            int,
            typer.Option()
        ] = 10,
):
    polyflexmd.data_analysis.utils.setup_logging()

    on_taurus = "taurus" in platform.node()
    _logger.info(f"On taurus: {on_taurus}")

    if on_taurus:
        job_extra_directives = [
            f'--job-name="polyflexmd-extract-fm-trajectory-process-worker"']
        if reservation is not None:
            job_extra_directives.append(f"--reservation={reservation}")
        cluster = SLURMCluster(
            queue=partition,
            cores=cores,
            processes=1,
            account=account,
            memory=memory,
            death_timeout=1800,
            walltime="48:00:00",
            local_directory="/tmp",
            interface="ib0",
            log_directory="/beegfs/ws/0/s4610340-polyflexmd/.logs",
            worker_extra_args=[f'--memory-limit "auto"'],
            job_extra_directives=job_extra_directives,

        )
        cluster.adapt(maximum_jobs=max_workers)
        client = dask.distributed.Client(cluster)
    else:
        client = dask.distributed.Client(n_workers=16, processes=True)

    _logger.info(client)
    _logger.info(f"Dashboard link: {client.dashboard_link}")

    _logger.info(f"{path_trajectories} exists => Reading processed ...")

    group_by_parameters = []

    if style.startswith("l_K"):
        group_by_parameters.append("kappa")
    if style == "l_K+d_end":
        group_by_parameters.append("d_end")

    traj_column_types = copy.deepcopy(polyflexmd.data_analysis.data.constants.RAW_TRAJECTORY_DF_COLUMN_TYPES)
    traj_column_types.pop("ix")
    traj_column_types.pop("iy")
    traj_column_types.pop("iz")
    traj_column_types["molecule-ID"] = np.uint16

    for param in group_by_parameters:
        traj_column_types[param] = "category"

    df_trajectories: dask.dataframe.DataFrame = dask.dataframe.read_csv(
        path_trajectories,
        dtype=traj_column_types,
    ).set_index("t", sort=False, sorted=True).persist()

    _logger.debug(f"N t partitions: {df_trajectories.npartitions}; t Divisions: {df_trajectories.divisions}")

    _logger.info("Extracting fm trajectory ...")
    df_fm_trajectory: pd.DataFrame = polyflexmd.data_analysis.transform.msdlm.extract_fm_trajectory_df(
        df_trajectories,
        group_by_columns=group_by_parameters,
        time_col="t"
    ).compute()

    _logger.info(f"Writing {output_path} ...")
    df_fm_trajectory.to_csv(output_path, index=True)


@app.command("calculate-msd")
def calculate_msd(
        path_monomer_traj: pathlib.Path,
        style: typing.Annotated[
            DataStyle,
            typer.Option(case_sensitive=False)
        ],
        output_path: typing.Annotated[
            pathlib.Path,
            typer.Option()
        ]
):
    polyflexmd.data_analysis.utils.setup_logging()

    group_by_parameters = []
    if style.startswith("l_K"):
        group_by_parameters.append("kappa")
    if style == "l_K+d_end":
        group_by_parameters.append("d_end")
    _logger.info(f"Reading monomer trajectory df {path_monomer_traj} ...")
    df_fm_trajectory = pd.read_csv(path_monomer_traj, index_col=[*group_by_parameters, "molecule-ID", "t"])

    _logger.info("Calculating msd of monomer ...")
    df_msd = polyflexmd.data_analysis.transform.msdlm.calculate_msd_lm_df(df_fm_trajectory, group_by_parameters,
                                                                          time_col="t")

    _logger.info(f"Writing msd of monomer to {output_path} ...")
    df_msd.to_csv(output_path, index=True)


@app.command(hidden=True)
def secret():
    raise NotImplementedError()


if __name__ == "__main__":
    app()
