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

    os.chdir("/work/")
    restrictions = []
    for config in configs.values():
        restr = config.get_restrictions()
        if len(restr) > level:
            restrictions.append(restr[level])
    try:
        train_table.populate(
            restrictions, display_progress=True, reserve_jobs=True, order=order
        )
    except LostConnectionError:
        raise LostConnectionError(
            "Connection to database lost at {}".format(datetime.now())
        )


def run_all_experiments(configs):
    from bias_transfer.tables.trained_model import TrainedModel, CollapsedTrainedModel
    from bias_transfer.tables.trained_transfer_model import (
        TrainedTransferModel,
        CollapsedTrainedTransferModel,
        TrainedTransferModel2,
    )

    run_experiments(configs, TrainedModel(), level=0)
    CollapsedTrainedModel().populate()
    run_experiments(configs, TrainedTransferModel(), level=1)
    CollapsedTrainedTransferModel().populate()
    run_experiments(configs, TrainedTransferModel2(), level=2)


def analyse(experiment, analysis_method, dataset="validation"):
    from bias_transfer.tables.trained_model import TrainedModel
    from bias_transfer.analysis.representation.noise_stability import (
        NoiseStabilityAnalyzer,
    )

    os.chdir("/work/")

    for desc, exp in experiment.experiments.items():
        if analysis_method == "all" or analysis_method == desc.name:
            analyser = NoiseStabilityAnalyzer(
                experiment=exp,
                name=desc.name,
                table=TrainedModel(),
                dataset=dataset,
                base_path="/work/analysis",
                num_samples=200,
                num_repeats=4,
                noise_std_max=0.51,
                noise_std_step=0.01,
            )
            analyser.run()


def main(experiment):
    fill_tables(experiment.transfer_experiments)
    run_all_experiments(experiment.transfer_experiments)


def checkout_and_install(
    repo_name, commit_hash, src_path="/src", checkout_path="/work", dev_mode=False
):
    path = os.path.join(src_path, repo_name)
    if not dev_mode:
        checkout_path = os.path.join(checkout_path, repo_name)
        if not os.path.exists(checkout_path):
            os.makedirs(checkout_path)
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


def load_experiment(
    recipe, experiment, schema=None, base_dir="./", import_prefix="", dev_mode=False
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
            "bias_transfer",
            "ml-utils",
            "nnfabrik",
            "nnvision",
            "pytorch_warmup",
        ):
            checkout_and_install(repo, "", dev_mode=True)
    else:
        with open(os.path.join(base_dir, recipe, "__commits.json"), "r") as read_file:
            commits_dict = json.load(read_file)
        experiment_commits = commits_dict.get(experiment, {})
        default_commits = commits_dict.get("default", {})
        for repo in (
            "bias_transfer",
            "ml-utils",
            "nnfabrik",
            "nnvision",
            "pytorch_warmup",
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
    dj.config["schema_name"] = (
        schema if schema else os.environ["DJ_USER"] + "_nnfabrik" + recipe
    )

    try:
        from bias_transfer.configs.base import Description
    except:
        from bias_transfer.experiments import Description  # legacy dependence
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
        "--dev-mode", dest="dev_mode", action="store_true", help="Use the current HEAD",
    )
    args = parser.parse_args()

    experiment = load_experiment(
        args.recipe, args.experiment, args.schema, dev_mode=args.dev_mode
    )

    if args.analysis:
        analyse(
            experiment, args.analysis, dataset=args.dataset,
        )
    else:
        main(experiment)
