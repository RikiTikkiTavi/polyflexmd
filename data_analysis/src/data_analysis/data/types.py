import typing

import pandas as pd
import pymatgen.io.lammps.data


class LammpsSystemData(typing.NamedTuple):
    box: pymatgen.io.lammps.data.LammpsBox
    masses: pd.DataFrame
    atoms: pd.DataFrame
    angles: pd.DataFrame
    bonds: pd.DataFrame
