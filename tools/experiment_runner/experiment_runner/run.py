import dataclasses
import logging
import pathlib
import shutil
import os
import subprocess

import experiment_runner.config
import git

import jinja2

_logger = logging.getLogger(__name__)


def run_experiment(experiment_config_path: pathlib.Path, clear_experiment_path: bool):
    _logger.info(f"Reading config from {experiment_config_path} ...")
    conf = experiment_runner.config.read_experiment_config(experiment_config_path)

    repo_root_path = pathlib.Path(__file__).parent.resolve()

    model_name = conf.simulation_model_path.stem
    repo = git.Repo(search_parent_directories=True)

    commit_sha = repo.git.rev_parse(repo.head.commit.hexsha, short=8)
    experiment_path = conf.experiments_path / model_name / commit_sha

    _logger.info(
        f"Deploying experiment: simulation={model_name} of version={commit_sha} into experiment_path={experiment_path} ..."
    )

    if clear_experiment_path:
        if experiment_path.exists():
            subprocess.run(f"rm -r {experiment_path}", shell=True)

    # Create experiment directory
    experiment_path.mkdir(parents=True, exist_ok=True)

    data_path = experiment_path / "data"
    data_path.mkdir()

    logs_path = experiment_path / "logs"
    logs_path.mkdir()

    checkpoints_path = experiment_path / "checkpoints"
    checkpoints_path.mkdir()

    shutil.copy(
        repo_root_path / conf.simulation_model_path,
        experiment_path / conf.simulation_model_path.name
    )

    # Copy experiment config into experiment dir
    shutil.copy(experiment_config_path, experiment_path / experiment_config_path.name)

    experiment_path_container = pathlib.Path(f"/experiment/{model_name}/{commit_sha}")

    logs_path_container = experiment_path_container / "logs"

    templates_path: pathlib.Path = pathlib.Path(
        f"{os.path.dirname(os.path.realpath(__file__))}/templates"
    )

    # Process system params

    if conf.system_config.name == "anchored-fene-chain":
        # noinspection PyDataclass
        kwargs = dataclasses.asdict(conf.system_config)
        kwargs.pop("name")
        system_params = dict(**kwargs, file_path=experiment_path_container / "initial_system.data")
    else:
        raise Exception(f"System {conf.system_config.name} is not supported by system-creator.")

    jinja_env: jinja2.Environment = jinja2.Environment(loader=jinja2.FileSystemLoader(templates_path))
    job_def: str = jinja_env.get_template("job.jinja2").render(
        {
            "job_params": {
                "name": f"{model_name}-{commit_sha}",
                "logs_path": logs_path_container,
                "account": conf.slurm_job_config.account,
                "time": conf.slurm_job_config.max_exec_time,
                "partition": conf.slurm_job_config.partition,
                "cpus_per_task": conf.slurm_job_config.cpus_per_task,
                "mem_per_cpu": conf.slurm_job_config.mem_per_cpu
            },
            "experiment_params": {
                "mount_path_host": experiment_path,
                "mount_path_container": experiment_path_container,
                "lammps_input_path_container": experiment_path_container / conf.simulation_model_path.name,
                "system_creator_simg": conf.system_creator_simg
            },
            "system_params": system_params
        }
    )

    path_to_job_file = experiment_path / "job_def.sh"

    path_to_job_file.write_text(job_def)

    subprocess.run(f"sbatch {path_to_job_file}", check=True, shell=True)
