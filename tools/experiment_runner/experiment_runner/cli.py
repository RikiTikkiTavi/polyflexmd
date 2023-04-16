import typer
import experiment_runner.run

app = typer.Typer()

app.command(name="run")(experiment_runner.run.run_experiment)

if __name__ == "__main__":
    app()
