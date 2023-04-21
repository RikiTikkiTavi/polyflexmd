import dataclasses
import pathlib
import typing
from typing import Generic, TypeVar

import numpy as np

_T = TypeVar("_T", int, float)


@dataclasses.dataclass
class Coordinates(Generic[_T]):
    x: _T
    y: _T
    z: _T

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "Coordinates":
        return Coordinates(*arr)

    def to_table_entry(self) -> str:
        return f"{self.x}  {self.y}  {self.z}"


class Monomer(typing.NamedTuple):
    id: int
    chain_id: int
    type: int
    r: Coordinates[float]
    ir: Coordinates[int]

    def to_table_entry(self) -> str:
        return f"{self.id}  {self.chain_id}  {self.type}  {self.r.to_table_entry()}  {self.ir.to_table_entry()}"


class Bond(typing.NamedTuple):
    id: int
    type: int
    monomers: tuple[Monomer, Monomer]

    def to_table_entry(self) -> str:
        return f"{self.id}  {self.type}  {self.monomers[0].id}  {self.monomers[1].id}"


class Angle(typing.NamedTuple):
    id: int
    type: int
    monomers: tuple[Monomer, Monomer, Monomer]

    def to_table_entry(self) -> str:
        return f"{self.id}  {self.type}  {self.monomers[0].id}  {self.monomers[1].id}  {self.monomers[2].id}"


class Chain(typing.NamedTuple):
    id: int
    monomers: list[Monomer]
    bonds: list[Bond]
    angles: list[Angle]


class Box(typing.NamedTuple):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def to_table_entry(self) -> str:
        lines = []
        for d in ("x", "y", "z"):
            lo, hi = getattr(self, d)
            lines.append(f"{lo} {hi} {d}lo {d}hi")
        return "\n".join(lines)


class AnchoredFENESystem(typing.NamedTuple):
    chains: list[Chain]
    box: Box


def calculate_monomer_type(monomer_ix: int, n_monomers: int, free_monomer_type: int):
    # Fixed monomers
    if monomer_ix <= 1:
        return 1

    # Last monomer in chain
    elif monomer_ix + 1 == n_monomers:
        return 3

    else:
        return free_monomer_type


def calculate_monomer_real_position(bond_length: float, prev_position: typing.Optional[Coordinates]) -> np.ndarray:
    if prev_position is None:
        return np.array([0., 0., 0.])
    else:
        phi = np.random.uniform(low=0, high=2*np.pi)
        theta = np.random.uniform(low=0, high=np.pi)
        rnd_vec_bond = np.array([
            np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi),
            np.cos(theta)
        ])
        return prev_position.to_numpy() + rnd_vec_bond


def apply_boundary_conditions(
        box_length: float, real_monomer_position: np.ndarray
) -> tuple[Coordinates[float], Coordinates[int]]:
    ir = np.zeros(3, dtype=int)
    dims = ["x", "y", "z"]
    s = box_length / 2

    for dim_i, dim_name in enumerate(dims):
        if abs(real_monomer_position[dim_i]) < s:
            ir[dim_i] = 0
        elif real_monomer_position[dim_i] > 0:
            ir[dim_i] = int(real_monomer_position[dim_i] // s)
        elif real_monomer_position[dim_i] < 0:
            ir[dim_i] = int(real_monomer_position[dim_i] // (-s))

    r = real_monomer_position - ir * box_length

    return Coordinates.from_numpy(r), Coordinates.from_numpy(ir)


def write_header(chains: list[Chain], box: Box, file: typing.TextIO):
    n_atoms = sum(len(c.monomers) for c in chains)
    n_bonds = sum(len(c.bonds) for c in chains)
    n_angles = sum(len(c.angles) for c in chains)

    header_lines = [
        '# LAMMPS FENE chain data file'
        '\n',
        f'{n_atoms}      atoms',
        f'{n_bonds}      bonds',
        '2      extra bond per atom',
        f'{n_angles}      angles',
        '0      dihedrals',
        '0      impropers',
        '3      atom types',
        '1      bond types',
        '1      angle types',
        '0      dihedral types',
        '0      improper types',
        '',
        box.to_table_entry(),
        '',
        'Masses',
        '',
        '1      1.000000',
        '2      1.000000',
        '3      1.000000'
    ]

    file.writelines(l + "\n" for l in header_lines)


def write_body(chains: list[Chain], file: typing.TextIO) -> None:
    file.write("\nAtoms\n\n")
    file.writelines(m.to_table_entry() + "\n" for c in chains for m in c.monomers)

    file.write("\nBonds\n\n")
    file.writelines(b.to_table_entry() + "\n" for c in chains for b in c.bonds)

    file.write("\nAngles\n\n")
    file.writelines(a.to_table_entry() + "\n" for c in chains for a in c.angles)


def dump_fene_bead_spring_system(system: AnchoredFENESystem, file_path: pathlib.Path) -> None:
    print(f"Writing output to {file_path} ...")
    with open(file_path, "w") as file:
        write_header(chains=system.chains, box=system.box, file=file)
        write_body(system.chains, file)


def create_fene_bead_spring_system(
        n_chains: int,
        n_monomers: int,
        monomer_type: int,
        bond_type: int,
        angle_type: int,
        bond_length: float,
        box_length: float,
        seed: int
) -> AnchoredFENESystem:
    np.random.seed(seed)

    chains: list[Chain] = []

    lo, hi = -box_length / 2, box_length / 2
    box = Box(x=(lo, hi), y=(lo, hi), z=(lo, hi))

    for chain_ix in range(n_chains):
        chain_id = chain_ix + 1

        monomers: list[Monomer] = []
        bonds: list[Bond] = []
        angles: list[Angle] = []

        for monomer_ix in range(n_monomers):
            monomer_id = chain_ix * n_monomers + monomer_ix + 1
            current_monomer_type = calculate_monomer_type(monomer_ix, n_monomers, monomer_type)

            monomer_r, monomer_ir = apply_boundary_conditions(
                box_length,
                real_monomer_position=calculate_monomer_real_position(
                    bond_length=bond_length,
                    prev_position=monomers[-1].r if len(monomers) > 0 else None
                )
            )

            current_monomer = Monomer(
                id=monomer_id,
                chain_id=chain_id,
                type=current_monomer_type,
                r=monomer_r,
                ir=monomer_ir
            )

            # Create bond
            if len(monomers) >= 1:
                bonds.append(Bond(
                    id=chain_ix * (n_monomers - 1) + len(bonds) + 1,
                    type=bond_type,
                    monomers=(monomers[-1], current_monomer),
                ))

            # Create angle
            if len(monomers) >= 2:
                angles.append(Angle(
                    id=chain_ix * (n_monomers - 2) + len(angles) + 1,
                    type=angle_type,
                    monomers=(monomers[-2], monomers[-1], current_monomer)
                ))

            monomers.append(current_monomer)

        chains.append(Chain(
            id=chain_id,
            monomers=monomers,
            bonds=bonds,
            angles=angles
        ))

    return AnchoredFENESystem(chains=chains, box=box)
