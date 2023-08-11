import itertools
import logging
import pathlib
import typing

import numpy as np
import pandas as pd

import polyflexmd.data_analysis.data.read as read
import polyflexmd.data_analysis.data.constants as constants
import polyflexmd.data_analysis.data.types
import polyflexmd.data_analysis.transform.transform as transform

import dask.dataframe

_logger = logging.getLogger(__name__)


def read_and_process_trajectories(
        trajectories: typing.Iterable[read.VariableTrajectoryPath],
        path_trajectories_interim: pathlib.Path,
        system: polyflexmd.data_analysis.data.types.LammpsSystemData,
        total_time_steps: int,
        time_steps_per_partition: int = 5,
) -> dask.dataframe.DataFrame:
    trajectories = list(trajectories)

    raw_column_types = constants.RAW_TRAJECTORY_DF_COLUMN_TYPES.copy()
    for var_name, var_val in trajectories[0].variables:
        raw_column_types[var_name] = "category"

    if not path_trajectories_interim.exists():
        _logger.info("Reformatting trajectories ...")

        path_trajectories_interim.mkdir(parents=True, exist_ok=True)

        polyflexmd.data_analysis.data.read.reformat_trajectories(
            vtps=trajectories,
            out_path=path_trajectories_interim,
            column_types=raw_column_types,
            chunks_per_file=time_steps_per_partition
        )

    df_trajectory_unfolded = transform.unfold_coordinates_df(
        trajectory_df=transform.join_raw_trajectory_df_with_system_data(
            raw_trajectory_df=read.read_lammps_trajectories(
                trajectories_interim_path=path_trajectories_interim,
                time_steps_per_partition=time_steps_per_partition,
                total_time_steps=total_time_steps,
                column_types=raw_column_types
            ),
            system_data=system
        ),
        system_data=system
    )

    return df_trajectory_unfolded.drop(["ix", "iy", "iz"], axis=1)
