import datetime
import pathlib
import git


def get_versions_info(experiments_path: pathlib.Path):
    repo = git.Repo(search_parent_directories=True)

    experiment_paths = experiments_path.glob("*")

    commits: list[git.Commit] = sorted(
        [repo.commit(p.stem) for p in experiments_path.glob("*")],
        key=lambda c: c.committed_datetime,
    )

    for i, commit in enumerate(commits):
        print(
            f"{i} - {repo.git.rev_parse(repo.head.commit.hexsha, short=8)} - {commit.committed_datetime} \n"
            f"{commit.message}"
            f"----------------"
        )
