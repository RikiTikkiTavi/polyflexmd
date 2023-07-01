import pathlib
import typing
from enum import Enum
import logging

import polyflexmd.data_analysis.pipelines.experiment
import typer

app = typer.Typer(invoke_without_command=False)


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
        ] = True
):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s :: %(message)s', datefmt='%d.%m.%Y %I:%M:%S'
    )

    polyflexmd.data_analysis.pipelines.experiment.process_experiment_data(
        path_experiment=path_experiment,
        n_workers=n_workers,
        style=style.value,
        read_relax=read_relax,
        enable_l_K_estimate=l_K_estimate
    )


@app.command(hidden=True)
def secret():
    raise NotImplementedError()


if __name__ == "__main__":
    app()
