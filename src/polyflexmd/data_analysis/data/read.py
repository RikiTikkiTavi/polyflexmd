import functools
import io
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import pathlib
import typing
import uuid

import numpy as np
import pandas as pd
import pymatgen.io.lammps.data
import polyflexmd.data_analysis.data.types as types
import polyflexmd.data_analysis.data.constants as constants

import dask.array
import dask.dataframe
import dask.bag

_logger = logging.getLogger(__name__)

import csv


def read_lammps_system_data(
        path: pathlib.Path,
        atom_style: str = "angle",
        molecule_id_type: np.dtype = np.ushort,
) -> types.LammpsSystemData:
    """
    Reads a LAMMPS data file and returns a dictionary with the header information
    and a pandas DataFrame with the atom coordinates and bonds information.
    """
    content = pymatgen.io.lammps.data.LammpsData.from_file(
        str(path),
        atom_style=atom_style,
        sort_id=False
    )

    content.atoms.rename({
        "nx": "ix",
        "ny": "iy",
        "nz": "iz"
    }, axis=1, inplace=True)

    content.atoms["molecule-ID"] = content.atoms["molecule-ID"].astype(molecule_id_type)

    return types.LammpsSystemData(
        box=content.box,
        masses=content.masses,
        atoms=content.atoms,
        angles=content.topology["Angles"],
        bonds=content.topology["Bonds"]
    )


def _read_atoms_step(
        file: typing.TextIO,
        particles_n: int,
        column_types: dict[str, typing.Any],
        columns: list[str],
        timestep: int
) -> typing.Generator[list[typing.Any], None, None]:
    for i in range(particles_n):
        # row = [
        #    column_types[col_name](raw_col_val)
        #    for col_name, raw_col_val in zip(columns, file.readline().split())
        # ]
        row = file.readline().split()
        row.insert(0, timestep)
        yield row


# https://gist.github.com/astyonax/1eb7b54326157299f0846324b5f1d98c
def read_lammps_custom_trajectory_file_generator(
        path: pathlib.Path,
        column_types: dict[str, typing.Any]
) -> typing.Generator[tuple[list[str], typing.Generator[list[typing.Any], None, None]], None, None]:
    with path.open('r') as file:

        line = file.readline()

        while line:

            if 'ITEM: TIMESTEP' in line:
                # begin new timestep
                timestep = int(file.readline())
                particles_n = 0

            if 'ITEM: NUMBER OF ATOMS' in line:
                particles_n = int(file.readline())

            if 'ITEM: ATOMS' in line:
                columns: list[str] = line.split()[2:]
                columns_n: int = len(columns)
                data_timestep = []
                if not (particles_n and columns_n):
                    raise StopIteration
                yield (
                    ["t", *columns],
                    _read_atoms_step(
                        file=file,
                        particles_n=particles_n,
                        column_types=column_types,
                        columns=columns,
                        timestep=timestep
                    )
                )

            line = file.readline()


class VariableTrajectoryPath(typing.NamedTuple):
    variables: list[tuple[str, float]]
    possible_values: list[list[float]]
    paths: list[pathlib.Path]


def process_timestep(df: dask.dataframe.DataFrame, path_to_var: dict[str, VariableTrajectoryPath]):
    timestep = df.iloc[1][0]
    columns = df.iloc[8][0].split()[2:]

    path = df.iloc[0]["path"]
    variable_names, variable_values = zip(*path_to_var[path].variables)

    header = ["t", *columns, *variable_names]
    rows = []

    for _, row in df.iloc[9:].iterrows():
        values = row["row"].split()
        values.insert(0, timestep)
        values.extend(variable_values)
        rows.append(values)

    return pd.DataFrame(rows, columns=header).astype(constants.RAW_TRAJECTORY_DF_COLUMN_TYPES)


def create_step_dataframe(lines: list[list[str]], columns: list[str]):
    return pd.DataFrame(data=lines, columns=columns)


def trajectory_to_timestep_dfs(
        path: pathlib.Path,
        variables: list[tuple[str, float]],
        out_path: pathlib.Path,
        file_prefix: str
):
    header_length = 9
    var_names, var_values = zip(*variables)
    var_values = list(var_values)

    with open(path, "r") as traj_file:

        n_timesteps = 0

        header = [next(traj_file) for _ in range(header_length)]

        t = int(header[1])

        n_timesteps += 1
        n_atoms = int(header[3])

        columns = header[8].split()[2:]
        columns = ["t"] + columns + list(var_names)

        while True:
            with open(out_path / f"{file_prefix}-{t}.csv", "w+", newline='') as chunk_file:
                writer = csv.writer(chunk_file, delimiter=";", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(columns)

                for i in range(n_atoms):
                    values = [t]
                    values.extend(next(traj_file).split())
                    values.extend(var_values)
                    writer.writerow(values)

            header = [next(traj_file) for _ in range(header_length)]

            t = int(header[1])

            n_timesteps += 1


def reformat_trajectories(
        vtps: list[VariableTrajectoryPath],
        out_path: pathlib.Path
):

    paths = [p for vtp in vtps for p in vtp.paths]
    args = [
        (p, vtp.variables, out_path, f"{i}_{j}")
        for i, vtp in enumerate(vtps) for j, p in enumerate(vtp.paths)
    ]

    with multiprocessing.pool.Pool(processes=len(paths)) as p:
        p.starmap(trajectory_to_timestep_dfs, args)


def read_lammps_trajectories(
        trajectories_interim_glob: str,
        column_types: dict = constants.RAW_TRAJECTORY_DF_COLUMN_TYPES,
        time_steps_per_partition: int = 100000,
        total_time_steps: typing.Optional[int] = None
) -> dask.dataframe.DataFrame:

    df = dask.dataframe.read_csv(trajectories_interim_glob, delimiter=";").persist()
    _logger.debug("Set index t ...")
    divisions = df["t"].loc[df["t"] % time_steps_per_partition == 0].drop_duplicates().compute().tolist()

    df = df.set_index("t", divisions=divisions)
    _logger.debug(f"N t partitions: {df.npartitions}; t Divisions: {df.divisions}")

    return df


def read_raw_trajectory_df(
        path: pathlib.Path,
        column_types: dict = constants.RAW_TRAJECTORY_DF_COLUMN_TYPES
):
    col_rows_iter_1, col_rows_iter_2 = itertools.tee(
        read_lammps_custom_trajectory_file_generator(path, column_types),
        2
    )
    columns = next(col_rows_iter_1)[0]
    rows = itertools.chain.from_iterable(rows_gen for _, rows_gen in col_rows_iter_2)
    return pd.DataFrame(data=rows, columns=columns, copy=False).astype(column_types)


def read_multiple_raw_trajectory_dfs(
        paths: list[pathlib.Path],
        column_types: dict = constants.RAW_TRAJECTORY_DF_COLUMN_TYPES,
        time_steps_per_partition: int = 100000,
) -> dask.dataframe.DataFrame:
    dfs = []

    for path in paths:
        dfs.append(
            read_lammps_trajectory(path, column_types=column_types, time_steps_per_partition=time_steps_per_partition))

    return dask.dataframe.concat(dfs)


def get_experiment_trajectories_paths(
        experiment_raw_data_path: pathlib.Path,
        style: typing.Literal["l_K+d_end", "l_K", "simple"],
        kappas: typing.Optional[list[float]] = None,
        d_ends: typing.Optional[list[float]] = None,
        continue_: bool = False,
        read_relax: bool = True
) -> typing.Generator[VariableTrajectoryPath, None, None]:
    suffix = "-continue" if continue_ else ""

    if style == "l_K+d_end":
        for i in range(1, len(kappas) + 1):
            for j in range(1, len(d_ends) + 1):
                p = experiment_raw_data_path / f"i_kappa={i}" / f"j_d_end={j}"
                paths_trajectories = [
                    p / f"polymer-{i}-{j}{suffix}.out"
                ]
                if read_relax:
                    paths_trajectories.insert(0, p / f"polymer_relax-{i}-{j}{suffix}.out")

                yield VariableTrajectoryPath(
                    variables=[("kappa", kappas[i - 1]), ("d_end", d_ends[j - 1])],
                    paths=paths_trajectories,
                    possible_values=[kappas, d_ends]
                )

    elif style == "l_K":
        for i in range(1, len(kappas) + 1):
            p = experiment_raw_data_path
            paths_trajectories = [
                p / f"polymer-{i}{suffix}.out"
            ]
            if read_relax:
                paths_trajectories.insert(0, p / f"polymer_relax-{i}{suffix}.out")

            yield VariableTrajectoryPath(
                variables=[("kappa", kappas[i - 1])],
                paths=paths_trajectories,
                possible_values=[kappas]
            )

    elif style == "simple":
        paths_trajectories = [
            experiment_raw_data_path / f"polymer.out"
        ]
        if read_relax:
            paths_trajectories.insert(0, experiment_raw_data_path / f"polymer_relax{suffix}.out")

        yield VariableTrajectoryPath(
            variables=[],
            paths=paths_trajectories,
            possible_values=[]
        )

    else:
        raise Exception(f"Unsupported style: {style}")
