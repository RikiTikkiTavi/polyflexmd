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
    for var_name, var_value in trajectory.variables:
        df_trajectory_unfolded[var_name] = var_value

    return df_trajectory_unfolded