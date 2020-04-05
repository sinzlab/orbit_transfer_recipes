"""
Initial setup based on https://github.com/kuangliu/pytorch-cifar
and https://github.com/weiaicunzai/pytorch-cifar100
"""
import json
from datetime import datetime
import os
import copy
from time import sleep

import argparse
import importlib
import subprocess
import sys
import site
from importlib import reload


def fill_tables(configs: dict):
    from nnfabrik.main import Fabrikant
    architect = dict(fabrikant_name=os.environ['USER'],
                     email=os.environ['EMAIL'],
                     affiliation=os.environ['AFFILIATION'],
                     dj_username=os.environ['DJ_USER'])
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
        train_table.populate(restrictions, display_progress=True, reserve_jobs=True, order=order)
    except LostConnectionError:
        raise LostConnectionError("Connection to database lost at {}".format(datetime.now()))


def run_all_experiments(configs):
    from bias_transfer.tables.trained_model import TrainedModel, CollapsedTrainedModel
    from bias_transfer.tables.trained_transfer_model import TrainedTransferModel, CollapsedTrainedTransferModel, \
        TrainedTransferModel2
    run_experiments(configs, TrainedModel(), level=0)
    CollapsedTrainedModel().populate()
    run_experiments(configs, TrainedTransferModel(), level=1)
    CollapsedTrainedTransferModel().populate()
    run_experiments(configs, TrainedTransferModel2(), level=2)


def analyse(experiment, experiment_key):
    dataset_cls = "CIFAR10"
    if experiment_key == "clean":
        exp = experiment.configs[Description(name=dataset_cls + ": Clean", seed=42)]
    if experiment_key == "noisy":
        exp = experiment.configs[Description(name=dataset_cls + ": Noise Augmented", seed=42)]
    if experiment_key == "rep_matching":
        exp = experiment.configs[Description(name=dataset_cls + ": Noise Augmented + Repr. Matching", seed=42)]
    if experiment_key == "adv_regression":
        exp = experiment.configs[Description(name=dataset_cls + ": Noise Augmented + Noise Adv Regession", seed=42)]

    # val_analyser = RepresentationAnalyser(experiment=exp, table=TrainedModel(), dataset="val",
    #                                       plot_style="lightpaper")
    # clean_indices = val_analyser.corr_matrix(mode="clean")
    # for i in range(1, 21):
    #     noise_level = 0.05 * i
    #     val_analyser.corr_matrix(mode="noisy", noise_level=noise_level)
    #     val_analyser.corr_matrix(mode="noisy", noise_level=noise_level, sorted_indices=clean_indices)
    #
    # del val_analyser
    train_analyser = RepresentationAnalyser(experiment=exp, table=TrainedModel(), dataset="train",
                                            plot_style="lightpaper")
    pca_clean = train_analyser.dim_reduction(noise_level=0.0, method="pca", mode="clean")
    for i in range(1, 11):
        noise_level = 0.05 * i
        train_analyser = RepresentationAnalyser(experiment=exp, table=TrainedModel(), dataset="train",
                                                plot_style="lightpaper")
        # train_analyser.dim_reduction(noise_level=noise_level, method="pca", mode="noisy")
        train_analyser.dim_reduction(noise_level=noise_level, method="pca", mode="noisy", pca=pca_clean)
    # train_analyser.dim_reduction(noise_level=0.0, method="tsne", mode="clean")
    # for i in range(1, 11):
    #     noise_level = 0.05 * i
    #     train_analyser = RepresentationAnalyser(experiment=exp, table=TrainedModel(), dataset="train",
    #                                             plot_style="lightpaper")
    #     train_analyser.dim_reduction(noise_level=noise_level, method="tsne", mode="noisy")


def main(experiment):
    fill_tables(experiment.transfer_experiments)
    run_all_experiments(experiment.transfer_experiments)


def checkout_and_install(repo_name, commit_hash, src_path="/src", checkout_path="/work", dev_mode=False):
    path = os.path.join(src_path, repo_name)
    if not dev_mode:
        checkout_path = os.path.join(checkout_path, repo_name)
        if not os.path.exists(checkout_path):
            os.makedirs(checkout_path)
        subprocess.check_call(["checkout_code", "--repository", path,
                               "--checkout-dir", checkout_path,
                               "--commit-hash", commit_hash])
        path = subprocess.check_output(["checkout_code", "--repository", path,
                                        "--checkout-dir", checkout_path,
                                        "--commit-hash", commit_hash,
                                        "--get-path"]).strip()  # run again to get path
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", path])
    reload(site)  # this will add it to sys.path
    sys.path.insert(1, sys.path.pop(-1))  # move it to front


def load_experiment(recipe, experiment, schema=None, base_dir="./", import_prefix="", dev_mode=False):
    if not recipe:
        sub_dirs = [dI for dI in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, dI))]
        recipe = sorted(sub_dirs)[0]
    if dev_mode:
        for repo in ("bias_transfer", "mlutils", "nnfabrik", "nnvision"):
            checkout_and_install(repo, "", dev_mode=True)
    else:
        with open(os.path.join(base_dir, recipe, "__commits.json"), "r") as read_file:
            commits_dict = json.load(read_file)
        experiment_commits = commits_dict.get(experiment)
        default_commits = commits_dict.get("default")
        for repo in ("bias_transfer", "mlutils", "nnfabrik", "nnvision"):
            commit_hash = experiment_commits.get(repo, default_commits.get(repo))
            if not commit_hash:
                raise LookupError("No commit hash found for repository {}".format(repo))
            checkout_and_install(repo, commit_hash, dev_mode=False)

    import datajoint as dj

    dj.config['database.host'] = os.environ['DJ_HOST']
    dj.config['database.user'] = os.environ['DJ_USER']
    dj.config['database.password'] = os.environ['DJ_PASS']
    dj.config['enable_python_native_blobs'] = True
    dj.config['schema_name'] = schema if schema else "anix_nnfabrik" + recipe

    try:
        from bias_transfer.configs.base import Description
    except:
        from bias_transfer.experiments import Description  # legacy dependence
    experiment_module = import_prefix + recipe + ("." + experiment if experiment else "")
    experiment = importlib.import_module(experiment_module)
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pre-defined recipes or analysis')
    parser.add_argument('--analysis', dest='analysis', action='store', default="", type=str,
                        help='name of experiment to analyse')
    parser.add_argument('--recipe', dest='recipe', action='store', default="", type=str,
                        help='set of recipes to run/analyse (set of experiments to execute)')
    parser.add_argument('--schema', dest='schema', action='store', default="", type=str,
                        help='schema in which to store recipes and results')
    parser.add_argument('--experiment', dest='experiment', action='store', default="", type=str,
                        help='name of experiment to run/analyse (specific experiment)')  # TODO find fitting name

    args = parser.parse_args()

    experiment = load_experiment(args.recipe, args.experiment, args.schema)

    from bias_transfer.analysis.representation_analysis import RepresentationAnalyser

    if args.analysis:
        analyse(experiment, args.analysis)
    else:
        main(experiment)
