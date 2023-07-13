import pathlib
import typing
from enum import Enum
import logging

import polyflexmd.data_analysis.pipelines.experiment
import typer

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
        save_angle_matrix: typing.Annotated[
            bool,
            typer.Option()
        ] = False,
        time_steps_per_partition: typing.Annotated[
            int,
            typer.Option()
        ] = 100000
):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s :: %(message)s', datefmt='%d.%m.%Y %I:%M:%S'
    )
    logging.getLogger("fsspec.local").setLevel(logging.INFO)
    logging.getLogger("distributed.scheduler").setLevel(logging.INFO)
    logging.getLogger("distributed.core").setLevel(logging.WARNING)
    logging.getLogger("distributed.nanny").setLevel(logging.WARNING)
    logging.getLogger("distributed.utils_perf").setLevel(logging.WARNING)

    import dask.distributed

    client = dask.distributed.Client(n_workers=n_workers, processes=True)

    _logger.info(client)

    polyflexmd.data_analysis.pipelines.experiment.process_experiment_data(
        path_experiment=path_experiment,
        n_workers=n_workers,
        style=style.value,
        read_relax=read_relax,
        enable_l_K_estimate=l_K_estimate,
        save_angle_matrix=save_angle_matrix,
        time_steps_per_partition=time_steps_per_partition
    )


@app.command(hidden=True)
def secret():
    raise NotImplementedError()


if __name__ == "__main__":
    app()
