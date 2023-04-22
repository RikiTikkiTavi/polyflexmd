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
                for i in range(particles_n):
                    line: str = file.readline()
                    data_timestep.append([timestep, *line.split()])
                timestep_df = pd.DataFrame(data=data_timestep, columns=["t", *columns]).astype(column_types)
                yield timestep_df

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
