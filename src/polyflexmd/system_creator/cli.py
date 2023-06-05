import pathlib

import typer as typer

import polyflexmd.system_creator as system_creator

app = typer.Typer()


@app.command(name="anchored-fene-chain")
def create_fene_bead_spring_system(
        file_path: pathlib.Path = typer.Option(..., "--file_path"),
        n_chains: int = typer.Option(..., "--n_chains"),
        n_monomers: int = typer.Option(..., "--n_monomers"),
        monomer_type: int = typer.Option(..., "--monomer_type"),
        bond_type: int = typer.Option(..., "--bond_type"),
        angle_type: int = typer.Option(..., "--angle_type"),
        bond_length: float = typer.Option(..., "--bond_length"),
        box_length: float = typer.Option(..., "--box_length"),
        seed: int = typer.Option(..., "--seed")
):
    print(f"Creating system of {n_chains} FENE beadspring chains, each consisting of {n_monomers} monomers of "
          f"type {monomer_type} in the middle of the chain, with {angle_type} angle type, with {bond_length} bond "
          f"lenth, with {bond_length} box length.")

    system = system_creator.anchored_fene_chain.create_fene_bead_spring_system(
        n_chains=n_chains,
        n_monomers=n_monomers,
        monomer_type=monomer_type,
        bond_type=bond_type,
        angle_type=angle_type,
        bond_length=bond_length,
        box_length=box_length,
        seed=seed
    )

    system_creator.anchored_fene_chain.dump_fene_bead_spring_system(system, file_path=file_path)


@app.command(name="anchored-fene-rod")
def create_fene_bead_spring_system_rod_like(
        file_path: pathlib.Path = typer.Option(..., "--file_path"),
        n_chains: int = typer.Option(..., "--n_chains"),
        n_monomers: int = typer.Option(..., "--n_monomers"),
        monomer_type: int = typer.Option(..., "--monomer_type"),
        bond_type: int = typer.Option(..., "--bond_type"),
        angle_type: int = typer.Option(..., "--angle_type"),
        bond_length: float = typer.Option(..., "--bond_length"),
        box_length: float = typer.Option(..., "--box_length"),
        seed: int = typer.Option(..., "--seed"),
        d_angle: float = typer.Option(0.0, "--d_angle")
):
    print(f"Creating system of {n_chains} FENE beadspring rod-like chains, each consisting of {n_monomers} monomers of "
          f"type {monomer_type} in the middle of the chain, with {angle_type} angle type, with {bond_length} bond "
          f"lenth, with {bond_length} box length.")

    system = system_creator.anchored_fene_rod.create_fene_bead_spring_system(
        n_chains=n_chains,
        n_monomers=n_monomers,
        monomer_type=monomer_type,
        bond_type=bond_type,
        angle_type=angle_type,
        bond_length=bond_length,
        box_length=box_length,
        seed=seed,
        d_angle=d_angle
    )

    system_creator.anchored_fene_chain.dump_fene_bead_spring_system(system, file_path=file_path)


if __name__ == "__main__":
    app()
