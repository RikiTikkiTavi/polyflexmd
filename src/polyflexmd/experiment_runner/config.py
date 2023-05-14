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
import tomlkit.items


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
    seed: int


@pydantic.dataclasses.dataclass
class SlurmJobConfig:
    account: str
    time: str
    partition: str
    nodes: int
    tasks_per_node: int
    ntasks: int
    cpus_per_task: int
    mem_per_cpu: int


@pydantic.dataclasses.dataclass
class SystemCreatorConfig:
    job: SlurmJobConfig
    venv_path: pathlib.Path
    system_config: AnchoredFENEChainConfig


@pydantic.dataclasses.dataclass
class SimulationConfig:
    job: SlurmJobConfig
    simulation_model_path: pathlib.Path
    experiments_path: pathlib.Path
    variables: typing.Optional[dict[str, Any]] = pydantic.Field(default_factory=dict)


@pydantic.dataclasses.dataclass
class ReportConfig:
    job: SlurmJobConfig
    venv_path: pathlib.Path
    notebook: pathlib.Path
    kernel: str
    notebook_params: dict


@pydantic.dataclasses.dataclass
class ExperimentConfig:
    simulation_config: SimulationConfig
    system_creator_config: SystemCreatorConfig
    report_config: typing.Optional[ReportConfig] = None


def read_experiment_config(config_file_path: pathlib.Path) -> ExperimentConfig:
    """
    Reads and validates experiment config from toml file
    :param config_file_path: path to config
    :return: config
    """
    with open(config_file_path) as conf_file:
        conf = tomlkit.loads(config_file_path.read_text()).unwrap()
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
        if section_content is not None:
            toml_config.append(section_name, section_content)

    with open(output_path, "w") as file:
        tomlkit.dump(toml_config, file)
