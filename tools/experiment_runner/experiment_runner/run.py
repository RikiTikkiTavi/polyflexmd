import dataclasses
import pathlib
import experiment_runner.config
import git
import system_creator


def run_experiment(experiment_config_path: pathlib.Path):
    conf = experiment_runner.config.read_experiment_config(experiment_config_path)
    model_name = conf.simulation_model_path.stem
    repo = git.Repo(search_parent_directories=True)
    commit_sha = repo.git.rev_parse(repo.head.commit.hexsha, short=8)
    experiment_path = conf.experiments_path / model_name / commit_sha

    # Create experiment directory
    experiment_path.mkdir(parents=True, exist_ok=True)

    data_path = experiment_path / "data"
    data_path.mkdir()
    logs_path = experiment_path / "logs"
    checkpoints_path = experiment_path / "checkpoints_path"

    if conf.system_config.name == "anchored-fene-chain":
        kwargs = dataclasses.asdict(conf.system_config)
        kwargs.pop("name")
        system_creator.anchored_fene_chain.create_fene_bead_spring_system(
            file_path=data_path / "initial_system.data",
            **kwargs
        )


run_experiment(pathlib.Path("/home/egor/Projects/bachelor-thesis/experiments/1-FENE-beadspring.toml"))
