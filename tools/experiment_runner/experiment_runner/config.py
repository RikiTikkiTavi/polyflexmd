import dataclasses
import io
import json
import pathlib
import typing
from datetime import datetime
from typing import Literal, Union, Any, TypeVar, Generic
import pydantic.dataclasses
import pydantic.json
import tomlkit
import subprocess
import git


@pydantic.dataclasses.dataclass
class AnchoredFENEChainConfig:
    name: str
    n_chains: int
    n_monomers: int
    monomer_type: int
    bond_type: int
    angle_type: int
    bond_length: float
    box_length: int


@pydantic.dataclasses.dataclass
class SlurmJobConfig:
    account: str
    max_exec_time: str
    partition: str
    cpus_per_task: int
    mem_per_cpu: int


@pydantic.dataclasses.dataclass
class ExperimentConfig:
    simulation_model_path: pathlib.Path
    experiments_path: pathlib.Path
    system_config: AnchoredFENEChainConfig
    slurm_job_config: SlurmJobConfig


def read_experiment_config(config_file_path: pathlib.Path) -> ExperimentConfig:
    """
    Reads and validates experiment config from toml file
    :param config_file_path: path to config
    :return: config
    """
    with open(config_file_path) as conf_file:
        conf = tomlkit.loads(config_file_path.read_text()).value
        return ExperimentConfig(**conf)


def write_experiment_config(config: ExperimentConfig, output_path: pathlib.Path, meta_data: bool = True):
    toml_config = tomlkit.document()

    if meta_data:
        repo = git.Repo(search_parent_directories=True)
        toml_config.add(tomlkit.comment(f"Commit: {repo.head.object.hexsha}"))
        toml_config.add(tomlkit.comment(f"Branch: {repo.active_branch}"))
        toml_config.add(tomlkit.comment(f"Dirty: {repo.is_dirty()}"))
        toml_config.add(tomlkit.comment(f"Datetime: {datetime.today()}"))

    for section_name, section_content in json.loads(json.dumps(config, default=pydantic.json.pydantic_encoder)).items():
        toml_config.append(section_name, section_content)

    with open(output_path, "w") as file:
        tomlkit.dump(toml_config, file)
