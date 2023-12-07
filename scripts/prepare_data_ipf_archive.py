import tarfile
import pathlib

directories_free_chain = [
    "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/17-EEA1_short-free_chain/ef6e4e76",
    "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/19-EEA1_short-lp_bonded_like-free_chain/ef6e4e76",
    "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K-vary-d_end/18-EEA1_short+Rab5_10x-free_chain/ef6e4e76",
    "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-full_flex/20-full_flex-free_chain/ef6e4e76"
]

directories_anchored_chain = [
    "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-vary-l_K/4-FENE-beadspring-vary-l_K/538accb2",
    "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-vary-l_K/14-EEA1_short/b7015f55",
    "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-vary-l_K-vary-d_end/15-EEA1_short+Rab5_10x/b7015f55",
    "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-vary-l_K-vary-d_end/16-EEA1_short+Rab5_20x/538accb2"
]

data_files_free_chain = [
    "data/processed/main_axis.csv",
    "data/processed/msd.csv",
    "data/processed/lm_msd.csv",
    "data/processed/lm_msd_avg-1000.csv",
    "data/processed/fm_msd.csv",
    "data/processed/fm_msd_avg.csv"
]

data_files_anchored_chain = [
    "data/processed/main_axis.csv",
    "data/processed/msd.csv",
]


def add_configs_and_scripts(arch_file: tarfile.TarFile, exp_dir: pathlib.Path):
    exp_conf = next(exp_dir.glob("*.toml"))
    print(f"Adding {exp_conf} ...")
    arch_file.add(exp_conf)

    lammps_file = next(exp_dir.glob("*.lammps"))
    arch_file.add(lammps_file)


with tarfile.open("/beegfs/ws/0/s4610340-polyflexmd/polyflexmd_results_ipf.tar.gz", "w:gz") as arch_file:
    for exp_dir in directories_free_chain:
        exp_dir = pathlib.Path(exp_dir)
        print(f"exp_dir={exp_dir}")

        add_configs_and_scripts(arch_file, exp_dir)

        for rel_file_path in data_files_free_chain:
            file_path = exp_dir / rel_file_path
            if file_path.exists():
                print(f"Adding {file_path} ...")
                arch_file.add(file_path)
            else:
                print(f"Does not exist: {file_path}")

    for exp_dir in directories_anchored_chain:
        exp_dir = pathlib.Path(exp_dir)
        print(f"exp_dir={exp_dir}")

        add_configs_and_scripts(arch_file, exp_dir)

        for rel_file_path in data_files_anchored_chain:
            file_path = exp_dir / rel_file_path
            print(f"Adding {file_path} ...")
            arch_file.add(file_path)
