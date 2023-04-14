import pathlib

import typer as typer

import system_creator.anchored_fene_chain

app = typer.Typer()
app.command(name="anchored-fene-chain")(system_creator.anchored_fene_chain.create_fene_bead_spring_system)


if __name__ == "__main__":
    app()
