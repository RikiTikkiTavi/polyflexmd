import numpy as np
import pathlib
import typing
from enum import Enum
import logging

import pandas as pd

import polyflexmd.data_analysis.transform.msdlm

import typer

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

    polyflexmd.data_analysis.transform.msdlm.calculate_msdlm_mean_avg_over_t_start(
        df_lm_traj=df_lm_traj,
        n_workers=n_workers,
        t_start=t_start,
        exclude_n_last=exclude_n_last,
        group_by_columns=variables,
        take_n_first=take_n_first
    ).to_csv(output_path, index=True)


@app.command(hidden=True)
def secret():
    raise NotImplementedError()


if __name__ == "__main__":
    app()
