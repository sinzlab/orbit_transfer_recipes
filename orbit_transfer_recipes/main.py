"""
Initial setup based on https://github.com/kuangliu/pytorch-cifar
and https://github.com/weiaicunzai/pytorch-cifar100
"""
import json
from datetime import datetime
import os

import argparse
import importlib
import subprocess
import sys
import site
from importlib import reload
from pathlib import Path
from filelock import Timeout, FileLock



def work_path(sub_path=""):
    path = os.environ.get("WORKDIR")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def src_path(sub_path=""):
    return os.environ.get("SRCDIR")


def fill_tables(configs: dict):
    from nnfabrik.main import Fabrikant

    architect = dict(
        fabrikant_name=os.environ["USER"],
        email=os.environ["EMAIL"],
        affiliation=os.environ["AFFILIATION"],
        dj_username=os.environ["DJ_USER"],
    )
    Fabrikant().insert1(architect, skip_duplicates=True)
    for config in configs.values():
        config.add_to_table()


def run_experiments(configs, train_table, order="random", level=0):
    from datajoint.errors import LostConnectionError
    from nntransfer.tables.transfer import TransferredTrainedModel
    from nntransfer.tables.nnfabrik import TrainedModelTransferRecipe

    os.chdir(work_path())
    restrictions = []
    for config in configs.values():
        restr = config.get_restrictions(level)
        if restr:
            restrictions.append(restr)
            if level > 0:  # add recipe
                transfer_from = config.get_restrictions(level - 1)
                transfer_to = config.configs[level].get_restrictions()[0]
                trainer_config = config.configs[level].trainer
                TrainedModelTransferRecipe().add_entry(
                    transfer_from=transfer_from,
                    transfer_to=transfer_to,
                    transfer_step=level,
                    data_transfer=trainer_config.data_transfer,
                )
                TransferredTrainedModel.transfer_recipe = [TrainedModelTransferRecipe()]
    if not restrictions:
        return False  # we've run all transfer steps
    try:
        train_table.populate(
            restrictions, display_progress=True, reserve_jobs=True, order=order
        )
    except LostConnectionError:
        raise LostConnectionError(
            "Connection to database lost at {}".format(datetime.now())
        )
    return True


def run_all_experiments(configs):
    from nntransfer.tables.transfer import TransferredTrainedModel

    level = 0
    while run_experiments(configs, TransferredTrainedModel(), level=level):
        level += 1


def main(experiment):
    fill_tables(experiment.transfer_experiments)
    run_all_experiments(experiment.transfer_experiments)


def checkout_and_install(
    repo_name,
    commit_hash,
    src_path=src_path(),
    checkout_path=work_path(),
    dev_mode=False,
):
    path = os.path.join(src_path, repo_name)
    checkout_path = os.path.join(checkout_path, repo_name)
    if not os.path.exists(checkout_path):
        os.makedirs(checkout_path)
    try:
        lock = FileLock(
            os.path.join(checkout_path, commit_hash + ".outer.lock"), timeout=30
        )
        with lock:
            if not dev_mode:
                subprocess.check_call(
                    [
                        "checkout_code",
                        "--repository",
                        path,
                        "--checkout-dir",
                        checkout_path,
                        "--commit-hash",
                        commit_hash,
                    ]
                )
                path = subprocess.check_output(
                    [
                        "checkout_code",
                        "--repository",
                        path,
                        "--checkout-dir",
                        checkout_path,
                        "--commit-hash",
                        commit_hash,
                        "--get-path",
                    ]
                ).strip()  # run again to get path
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", path])
            reload(site)  # this will add it to sys.path
            sys.path.insert(1, sys.path.pop(-1))  # move it to front
    except Timeout:
        print("Could not acquire lock. Timeout.")


def load_experiment(
    recipe,
    experiment,
    schema=None,
    base_dir="./bias_transfer_recipes",
    import_prefix="",
    dev_mode=False,
):
    if not recipe:
        sub_dirs = [
            dI
            for dI in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, dI))
        ]
        recipe = sorted(sub_dirs)[0]
    if dev_mode:
        for repo in (
            "orbit_transfer",
            "nntransfer",
            "neuralpredictors",
            "nnfabrik",
            "pytorch_warmup"
        ):
            checkout_and_install(repo, "", dev_mode=True)
    else:
        with open(os.path.join(base_dir, recipe, "__commits.json"), "r") as read_file:
            commits_dict = json.load(read_file)
        experiment_commits = commits_dict.get(experiment, {})
        default_commits = commits_dict.get("default", {})
        for repo in (
            "orbit_transfer",
            "nntransfer",
            "neuralpredictors",
            "nnfabrik",
            "pytorch_warmup"
        ):
            commit_hash = experiment_commits.get(repo, default_commits.get(repo))
            if not commit_hash:
                raise LookupError("No commit hash found for repository {}".format(repo))
            checkout_and_install(repo, commit_hash, dev_mode=False)

    import datajoint as dj

    dj.config["database.host"] = os.environ["DJ_HOST"]
    dj.config["database.user"] = os.environ["DJ_USER"]
    dj.config["database.password"] = os.environ["DJ_PASS"]
    dj.config["enable_python_native_blobs"] = True
    if not "stores" in dj.config:
        dj.config["stores"] = {}
    dj.config["stores"]["minio"] = {  # store in s3
        "protocol": "s3",
        "endpoint": os.environ.get("MINIO_ENDPOINT", "DUMMY_ENDPOINT"),
        "bucket": "nnfabrik",
        "location": "dj-store",
        "access_key": os.environ.get("MINIO_ACCESS_KEY", "FAKEKEY"),
        "secret_key": os.environ.get("MINIO_SECRET_KEY", "FAKEKEY"),
        'secure': True,
    }
    dj.config["custom"] = {}
    dj.config["custom"]["nnfabrik.my_schema_name"] = (
        schema if schema else f"bias_transfer_{recipe}"
    )

    experiment_module = (
        import_prefix + recipe + ("." + experiment if experiment else "")
    )
    experiment = importlib.import_module(experiment_module)
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running pre-defined recipes or analysis"
    )
    parser.add_argument(
        "--recipe",
        dest="recipe",
        action="store",
        default="",
        type=str,
        help="set of recipes to run/analyse (set of experiments to execute)",
    )
    parser.add_argument(
        "--schema",
        dest="schema",
        action="store",
        default="",
        type=str,
        help="schema in which to store recipes and results",
    )
    parser.add_argument(
        "--experiment",
        dest="experiment",
        action="store",
        default="",
        type=str,
        help="name of experiment to run/analyse (specific experiment)",
    )
    parser.add_argument(
        "--analysis",
        dest="analysis",
        action="store",
        default="",
        type=str,
        help="analysis method to execute",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        action="store",
        default="validation",
        type=str,
        help="dataset to perform analysis on",
    )
    parser.add_argument(
        "--dev-mode",
        dest="dev_mode",
        action="store_true",
        help="Use the current HEAD",
    )
    args = parser.parse_args()

    experiment = load_experiment(
        args.recipe, args.experiment, args.schema, dev_mode=args.dev_mode
    )

    main(experiment)
