import pathlib
import typing
from enum import Enum
import logging

import polyflexmd.data_analysis.pipelines.experiment
import typer

import platform

import dask.distributed
from dask_jobqueue import SLURMCluster

app = typer.Typer(invoke_without_command=False)

_logger = logging.getLogger(__name__)


class DataStyle(str, Enum):
    l_K = "l_K"
    l_K_d_end = "l_K+d_end"
    simple = "simple"


@app.command("process-experiment-data")
def process_experiment_data(
        path_experiment: pathlib.Path,
        style: typing.Annotated[
            DataStyle,
            typer.Option(case_sensitive=False)
        ],
        n_workers: typing.Annotated[
            int,
            typer.Option()
        ] = 16,
        read_relax: typing.Annotated[
            bool,
            typer.Option(),
        ] = False,
        l_K_estimate: typing.Annotated[
            bool,
            typer.Option()
        ] = True,
        time_steps_per_partition: typing.Annotated[
            int,
            typer.Option()
        ] = 5,
        total_time_steps: typing.Annotated[
            typing.Optional[int],
            typer.Option()
        ] = None,
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
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s :: %(message)s', datefmt='%d.%m.%Y %I:%M:%S'
    )
    logging.getLogger("fsspec.local").setLevel(logging.INFO)
    logging.getLogger("distributed.scheduler").setLevel(logging.WARNING)
    logging.getLogger("distributed.core").setLevel(logging.WARNING)
    logging.getLogger("distributed.nanny").setLevel(logging.WARNING)
    logging.getLogger("distributed.utils_perf").setLevel(logging.WARNING)
    logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
    logging.getLogger("dask_jobqueue.core").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    on_taurus = "taurus" in platform.node()
    _logger.info(f"On taurus: {on_taurus}")

    if on_taurus:
        job_extra_directives = [f'--job-name="polyflexmd-f{path_experiment.parents[0].stem}-{path_experiment.stem}-process-worker"']
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
            worker_extra_args=[f"--memory-limit auto"],
            job_extra_directives=job_extra_directives,

        )
        cluster.adapt(maximum_jobs=max_workers)
        client = dask.distributed.Client(cluster)
    else:
        client = dask.distributed.Client(n_workers=n_workers, processes=True)

    _logger.info(client)
    _logger.info(f"Dashboard link: {client.dashboard_link}")

    polyflexmd.data_analysis.pipelines.experiment.process_experiment_data(
        path_experiment=path_experiment,
        style=style.value,
        read_relax=read_relax,
        enable_l_K_estimate=l_K_estimate,
        calculate_msd_lm=True,
        calculate_lm_trajectory=True,
        time_steps_per_partition=time_steps_per_partition,
        total_time_steps=total_time_steps
    )


@app.command(hidden=True)
def secret():
    raise NotImplementedError()


if __name__ == "__main__":
    app()
