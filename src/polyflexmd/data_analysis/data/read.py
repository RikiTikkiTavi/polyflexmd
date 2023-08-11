import functools
import io
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import pathlib
import re
import typing
import uuid

import numpy as np
import pandas
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
        file_prefix: str,
        column_types: dict,
        chunks_per_file: int = 5
):
    _logger.info(f"Reformatting {path} ...")

    header_length = 9
    if len(variables) > 0:
        var_names, var_values = zip(*variables)
    else:
        var_names = []
        var_values = []
    var_values = [str(v) for v in var_values]

    with open(path, "r") as traj_file:
        header = [next(traj_file) for _ in range(header_length)]
        t = int(header[1])
        n_atoms = int(header[3])

        raw_columns = header[8].split()[2:]
        out_columns = ["t"] + raw_columns + list(var_names)

        lines = []
        n_chunks = 0
        t_first = t

        while True:
            for i in range(n_atoms):
                row = [t]
                row.extend(next(traj_file).split())
                row.extend(var_values)
                lines.append(row)

            n_chunks += 1
            last_chunk = n_chunks == chunks_per_file

            if last_chunk:
                if t % 1e6 == 0:
                    _logger.debug(f"Writing chunk {t_first}-{t} of {path} ...")

                pd.DataFrame(
                    lines, columns=out_columns
                ).to_csv(
                    out_path / f"{file_prefix}-{t_first}-{t}.csv", index=False
                )

                lines.clear()
                n_chunks = 0

            line = traj_file.readline()

            if line == "":
                pd.DataFrame(
                    lines, columns=out_columns
                ).to_csv(
                    out_path / f"{file_prefix}-{t_first}-{t}.csv", index=False
                )
                break

            header = [next(traj_file) for _ in range(header_length - 1)]
            t = int(header[0])

            if last_chunk:
                t_first = t


def reformat_trajectories(
        vtps: list[VariableTrajectoryPath],
        out_path: pathlib.Path,
        column_types: dict
):
    paths = [p for vtp in vtps for p in vtp.paths]
    args = [
        (p, vtp.variables, out_path, f"{i}_{j}", column_types)
        for i, vtp in enumerate(vtps) for j, p in enumerate(vtp.paths)
    ]

    with multiprocessing.pool.Pool(processes=len(paths)) as p:
        p.starmap(trajectory_to_timestep_dfs, args)


def read_lammps_trajectories(
        trajectories_interim_path: pathlib.Path,
        total_time_steps: int,
        column_types: dict,
        time_steps_per_partition: int = 100000,
) -> dask.dataframe.DataFrame:
    trajectories_interim_glob = str(trajectories_interim_path / "*.csv")

    divisions = sorted((
        int(re.search(r"0_0-(\d+)-\d+.csv", timestep_path.name).groups()[0])
        for timestep_path in trajectories_interim_path.glob("0_0-*.csv")
    ))
    if divisions[-1] != total_time_steps:
        divisions.append(total_time_steps)

    _logger.info("Compute divisions ...")

    _logger.debug("Read and set index t ...")
    df = dask.dataframe.read_csv(
        trajectories_interim_glob
    ).set_index("t").repartition(divisions=divisions).persist()

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
