import itertools
import pathlib
import typing

import numpy as np
import pandas as pd

import polyflexmd.data_analysis.data.read as read
import polyflexmd.data_analysis.data.types
import polyflexmd.data_analysis.transform.transform as transform


def read_and_process_trajectory(
        trajectory: read.VariableTrajectoryPath,
        system: polyflexmd.data_analysis.data.types.LammpsSystemData
):
    df_trajectory_unfolded = transform.unfold_coordinates_df(
        trajectory_df=transform.join_raw_trajectory_df_with_system_data(
            raw_trajectory_df=read.read_multiple_raw_trajectory_dfs(trajectory.paths),
            system_data=system
        ),
        system_data=system
    )

    df_trajectory_unfolded.drop(["ix", "iy", "iz"], axis=1, inplace=True)

    for (var_name, var_value), var_possible_values in zip(trajectory.variables, trajectory.possible_values):
        df_trajectory_unfolded[var_name] = pd.Categorical(
            values=np.repeat(var_value, len(df_trajectory_unfolded)),
            categories=var_possible_values,
        )

    return df_trajectory_unfolded


def read_and_process_trajectories(
        trajectories: typing.Iterable[read.VariableTrajectoryPath],
        system: polyflexmd.data_analysis.data.types.LammpsSystemData
):
    return pd.concat([
        read_and_process_trajectory(path, system)
        for path in trajectories
    ])


