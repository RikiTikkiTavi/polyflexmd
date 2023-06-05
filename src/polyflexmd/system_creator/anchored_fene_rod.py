import typing

from polyflexmd.system_creator.anchored_fene_chain import (
    AnchoredFENESystem,
    Monomer,
    apply_boundary_conditions,
    calculate_monomer_type,
    Bond,
    Angle,
    Chain,
    Box,
    Coordinates
)
import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_monomer_real_position(
        bond_length: float,
        prev_positions: tuple[typing.Optional[Coordinates], typing.Optional[Coordinates]],
        d_angle: float = 0.0
) -> np.ndarray:
    # No prev monomer => 0
    if prev_positions[1] is None:
        return np.array([0., 0., 0.])
    # No second last monomer => this is second monomer
    elif prev_positions[0] is None:
        phi = np.random.uniform(low=0, high=2 * np.pi)
        theta = np.random.uniform(low=0, high=np.pi)
        rnd_vec_bond = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ]) * bond_length
        return prev_positions[1].to_numpy() + rnd_vec_bond
    # At least 3 monomer
    else:
        prev_bond_vec = prev_positions[1].to_numpy() - prev_positions[0].to_numpy()
        rot_angles = np.random.uniform(low=-d_angle, high=d_angle, size=3)
        rotation = R.from_euler('xyz', rot_angles, degrees=True)
        return prev_positions[1].to_numpy() + rotation.apply(prev_bond_vec)


def create_fene_bead_spring_system(
        n_chains: int,
        n_monomers: int,
        monomer_type: int,
        bond_type: int,
        angle_type: int,
        bond_length: float,
        box_length: float,
        seed: int,
        d_angle: float = 0.0
) -> AnchoredFENESystem:
    np.random.seed(seed)

    chains: list[Chain] = []

    lo, hi = -box_length / 2, box_length / 2
    box = Box(x=(lo, hi), y=(lo, hi), z=(lo, hi))

    for chain_ix in range(n_chains):
        chain_id = chain_ix + 1

        monomers: list[Monomer] = []
        monomers_real_coords: list[Coordinates] = []
        bonds: list[Bond] = []
        angles: list[Angle] = []

        for monomer_ix in range(n_monomers):
            monomer_id = chain_ix * n_monomers + monomer_ix + 1
            current_monomer_type = calculate_monomer_type(monomer_ix, n_monomers, monomer_type)

            monomer_real_r: np.ndarray = calculate_monomer_real_position(
                bond_length=bond_length,
                prev_positions=(
                    monomers_real_coords[-2] if len(monomers_real_coords) > 1 else None,
                    monomers_real_coords[-1] if len(monomers_real_coords) > 0 else None
                ),
                d_angle=d_angle
            )

            monomers_real_coords.append(Coordinates.from_numpy(monomer_real_r))

            monomer_r, monomer_ir = apply_boundary_conditions(
                box_length,
                real_monomer_position=monomer_real_r
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
