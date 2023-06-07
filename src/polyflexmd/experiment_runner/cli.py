import pathlib

import typer
import polyflexmd.experiment_runner.run
import polyflexmd.experiment_runner.versions_info

app = typer.Typer()


@app.command(name="run")
def run_experiment(
        experiment_config_path: pathlib.Path,
        clear_experiment_path: bool = typer.Option(False, "--clear_experiment_path")
):
    polyflexmd.experiment_runner.run.run_experiment(experiment_config_path, clear_experiment_path)


@app.command(name="versions-info")
def versions_info(experiments_path: pathlib.Path):
    polyflexmd.experiment_runner.versions_info.get_versions_info(experiments_path)


if __name__ == "__main__":
    app()
