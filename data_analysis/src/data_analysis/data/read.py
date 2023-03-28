import pathlib
import typing
import pandas as pd
from data_analysis.data.types import LammpsSystemTopology

def read_lammps_system_topology_file(
        path: pathlib.Path,
        atom_style_columns: list[str]
) -> LammpsSystemTopology:
    pass


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
                columns = []

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
