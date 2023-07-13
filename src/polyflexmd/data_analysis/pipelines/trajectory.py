import itertools
import logging
import pathlib
import typing

import numpy as np
import pandas as pd

import polyflexmd.data_analysis.data.read as read
import polyflexmd.data_analysis.data.types
import polyflexmd.data_analysis.transform.transform as transform

import dask.dataframe

_logger = logging.getLogger(__name__)


def read_and_process_trajectories(
        trajectories: typing.Iterable[read.VariableTrajectoryPath],
        system: polyflexmd.data_analysis.data.types.LammpsSystemData,
        time_steps_per_partition: int = 100000
) -> dask.dataframe.DataFrame:
    df_trajectory_unfolded = transform.unfold_coordinates_df(
        trajectory_df=transform.join_raw_trajectory_df_with_system_data(
            raw_trajectory_df=read.read_lammps_trajectories(
                list(trajectories),
                time_steps_per_partition=time_steps_per_partition
            ),
            system_data=system
        ),
        system_data=system
    )

    return df_trajectory_unfolded.drop(["ix", "iy", "iz"], axis=1)
