import dataclasses
import logging
import pathlib
import shutil
import os
import subprocess

import polyflexmd.experiment_runner.config
import git

import jinja2

_logger = logging.getLogger(__name__)


# noinspection PyDataclass
def run_experiment(experiment_config_path: pathlib.Path, clear_experiment_path: bool):
    logging.basicConfig()
    _logger.setLevel(logging.DEBUG)

    _logger.info(f"Reading config from {experiment_config_path} ...")
    conf = polyflexmd.experiment_runner.config.read_experiment_config(experiment_config_path)

    repo_root_path = pathlib.Path(__file__).parents[3].resolve()
    _logger.debug(f"Repo root path resolved to {repo_root_path}")

    model_name = conf.simulation_config.simulation_model_path.stem
    repo = git.Repo(search_parent_directories=True)

    commit_sha = repo.git.rev_parse(repo.head.commit.hexsha, short=8)
    experiment_path = conf.simulation_config.experiments_path / model_name / commit_sha

    _logger.info(
        f"Deploying experiment: simulation={model_name} of VERSION={commit_sha} into experiment_path={experiment_path} ..."
    )

    if clear_experiment_path:
        if experiment_path.exists():
            _logger.info(f"Clearing workdir {experiment_path} ...")
            subprocess.run(f"rm -r {experiment_path}", shell=True)

    _logger.info(f"Creating directories ...")
    # Create experiment directory
    experiment_path.mkdir(parents=True, exist_ok=True)

    data_path = experiment_path / "data"
    data_path.mkdir()

    logs_path = experiment_path / "logs"
    logs_path.mkdir()

    checkpoints_path = experiment_path / "checkpoints"
    checkpoints_path.mkdir()

    shutil.copy(
        repo_root_path / conf.simulation_config.simulation_model_path,
        experiment_path / conf.simulation_config.simulation_model_path.name
    )

    # Copy experiment config into experiment dir
    shutil.copy(experiment_config_path, experiment_path / experiment_config_path.name)

    experiment_path_container = pathlib.Path(f"/experiment/{model_name}")

    logs_path_container = experiment_path_container / "logs"

    templates_path: pathlib.Path = pathlib.Path(
        f"{os.path.dirname(os.path.realpath(__file__))}/templates"
    )

    # Process system params

    if conf.system_creator_config.system_config.name == "anchored-fene-chain":
        # noinspection PyDataclass
        kwargs = dataclasses.asdict(conf.system_creator_config.system_config)
        kwargs.pop("name")
        system_params = dict(**kwargs, file_path=experiment_path / "data" / "initial_system.data")
    else:
        raise Exception(f"System {conf.system_creator_config.system_config.name} is not supported by system-creator.")

    jinja_env: jinja2.Environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_path),
        comment_start_string='{=',
        comment_end_string='=}',
    )

    job_system_creator = jinja_env.get_template("job_system_creator.jinja2").render({
        "system_creator": {
            "job": {
                "name": f"{model_name}-create_system-{commit_sha}",
                "logs_path": logs_path,
                **dataclasses.asdict(conf.system_creator_config.job)
            },
            "system_params": system_params,
            "venv_path": conf.report_config.venv_path
        }
    })

    job_simulation = jinja_env.get_template("job_run_simulation.jinja2").render({
        "simulation": {
            "job": {
                "name": f"{model_name}-{commit_sha}",
                "logs_path": logs_path,
                **dataclasses.asdict(conf.simulation_config.job)
            },
            "mount_path_host": experiment_path,
            "mount_path_container": experiment_path_container,
            "logs_path": logs_path_container,
            "lammps_input_path_container": experiment_path_container / conf.simulation_config.simulation_model_path.name
        }
    })

    job_report = jinja_env.get_template("job_generate_report.jinja2").render({
        "report": {
            "job": {
                "name": f"{model_name}-report-{commit_sha}",
                "logs_path": logs_path,
                **dataclasses.asdict(conf.simulation_config.job)
            },
            "venv_path": conf.report_config.venv_path,
            "input_notebook": repo_root_path / conf.report_config.notebook,
            "output_notebook": experiment_path / conf.report_config.notebook,
            "report_name": conf.report_config.notebook.name,
            "report_dir": experiment_path,
            "kernel": conf.report_config.kernel,
            "notebook_params": {
                "PATH_EXPERIMENT": experiment_path,
                **conf.report_config.notebook_params
            }
        }
    })

    job_file_names = ["job_system_creator.sh", "job_run_simulation.sh", "job_generate_report.sh"]
    for job_file_name, job_def in zip(job_file_names, [job_system_creator, job_simulation, job_report]):
        path_to_job_file = experiment_path / job_file_name
        path_to_job_file.write_text(job_def)

    chain_job_submit_script = jinja_env.get_template("submit_chain_jobs.jinja2").render({
        "job_files": [str(experiment_path / name) for name in job_file_names]
    })
    chain_job_submit_script_path = experiment_path / "submit_jobs.sh"
    chain_job_submit_script_path.write_text(chain_job_submit_script)

    subprocess.run(f"chmod +x {chain_job_submit_script_path}", shell=True, check=True)
    subprocess.call(str(chain_job_submit_script_path), shell=True)
