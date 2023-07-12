import io
import itertools
import logging
import pathlib
import typing

import numpy as np
import pandas as pd
import pymatgen.io.lammps.data
import polyflexmd.data_analysis.data.types as types
import polyflexmd.data_analysis.data.constants as constants

import dask.array
import dask.dataframe
import dask.bag

_logger = logging.getLogger(__name__)

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


def process_timestep(df: dask.dataframe.DataFrame):
    timestep = df.iloc[1][0]
    columns = df.iloc[8][0].split()[2:]

    header = ["t", *columns]
    rows = []
    for _, row in df.iloc[9:].to_records(index=True):
        values = row.split()
        values.insert(0, timestep)
        rows.append(values)

    return pd.DataFrame(rows, columns=header).astype(constants.RAW_TRAJECTORY_DF_COLUMN_TYPES)


def read_lammps_trajectory(
        path: pathlib.Path,
        column_types: dict = constants.RAW_TRAJECTORY_DF_COLUMN_TYPES
) -> dask.dataframe.DataFrame:
    _logger.debug(f"Reading {path}...")
    df_bag = dask.bag.read_text(str(path), linedelimiter="\n", blocksize="128MiB").to_dataframe(columns=["row"])
    _logger.debug(f"Extracting columns from {path}...")
    columns = df_bag.loc[df_bag["row"].str.contains("ITEM: ATOMS")].head(1).iloc[0]["row"].split()[2:]
    columns.insert(0, "t")
    _logger.debug(f"Creating dataframe from {path}...")
    grouper = df_bag["row"].str.contains("ITEM: TIMESTEP").cumsum()
    groups_count = len(grouper.unique().compute())
    df_groups = df_bag.groupby(grouper)
    df: dask.dataframe.DataFrame = df_groups.apply(
        process_timestep,
        meta=pd.DataFrame(columns=columns).astype(column_types)
    ).reset_index(drop=True).repartition(npartitions=groups_count // 10)
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
        column_types: dict = constants.RAW_TRAJECTORY_DF_COLUMN_TYPES
) -> dask.dataframe.DataFrame:
    dfs = []

    for path in paths:
        dfs.append(read_lammps_trajectory(path, column_types=column_types))

    return dask.dataframe.concat(dfs)


class VariableTrajectoryPath(typing.NamedTuple):
    variables: list[tuple[str, float]]
    possible_values: list[list[float]]
    paths: list[pathlib.Path]


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
