import io
import itertools
import pathlib
import typing
import pandas as pd
import pymatgen.io.lammps.data
import polyflexmd.data_analysis.data.types as types
import polyflexmd.data_analysis.data.constants as constants


def read_lammps_system_data(
        path: pathlib.Path,
        atom_style: str = "angle"
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
        row = [
            column_types[col_name](raw_col_val)
            for col_name, raw_col_val in zip(columns, file.readline().split())
        ]
        row.insert(0, timestep)
        yield row


# https://gist.github.com/astyonax/1eb7b54326157299f0846324b5f1d98c
def read_lammps_custom_trajectory_file(
        path: pathlib.Path,
        column_types: dict[str, typing.Any]
) -> typing.Generator[pd.DataFrame, None, None]:
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
                yield pd.DataFrame(
                    data=_read_atoms_step(
                        file=file,
                        particles_n=particles_n,
                        column_types=column_types,
                        columns=columns,
                        timestep=timestep
                    ),
                    columns=["t", *columns]
                )

            line = file.readline()


def read_raw_trajectory_df(
        path: pathlib.Path,
        column_types: dict = constants.RAW_TRAJECTORY_DF_COLUMN_TYPES
):
    return pd.concat(read_lammps_custom_trajectory_file(
        path=path,
        column_types=column_types
    ))


def read_multiple_raw_trajectory_dfs(
        paths: list[pathlib.Path],
        column_types: dict = constants.RAW_TRAJECTORY_DF_COLUMN_TYPES
):
    return pd.concat(
        itertools.chain.from_iterable(
            read_lammps_custom_trajectory_file(
                path=path,
                column_types=column_types
            ) for path in paths
        )
    )


class VariableTrajectoryPath(typing.NamedTuple):
    variables: list[tuple[str, float]]
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
                    variables=[("kappa", kappas[i - 1]), ("d_end", d_ends[i - 1])],
                    paths=paths_trajectories
                )

    elif style == "l_K":
        for i in range(1, len(kappas) + 1):
            p = experiment_raw_data_path / f"i_kappa={i}"
            paths_trajectories = [
                p / f"polymer-{i}{suffix}.out"
            ]
            if read_relax:
                paths_trajectories.insert(0, p / f"polymer_relax-{i}{suffix}.out")

            yield VariableTrajectoryPath(
                variables=[("kappa", kappas[i - 1])],
                paths=paths_trajectories
            )

    elif style == "simple":
        paths_trajectories = [
            experiment_raw_data_path / f"polymer.out"
        ]
        if read_relax:
            paths_trajectories.insert(0, experiment_raw_data_path / f"polymer_relax{suffix}.out")

        yield VariableTrajectoryPath(
            variables=[],
            paths=paths_trajectories
        )

    else:
        raise Exception(f"Unsupported style: {style}")
