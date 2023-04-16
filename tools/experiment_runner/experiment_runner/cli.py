import pathlib

import typer
import experiment_runner.run

app = typer.Typer()


@app.command(name="run")
def run_experiment(
        experiment_config_path: pathlib.Path,
        clear_experiment_path: bool = typer.Option(False, "--clear_experiment_path")
):
    experiment_runner.run.run_experiment(experiment_config_path, clear_experiment_path)


if __name__ == "__main__":
    app()
