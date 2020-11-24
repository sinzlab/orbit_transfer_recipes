from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}


for dataset_cls in (
    # "CIFAR100",
    # "CIFAR10",
    "TinyImageNet",
    # "ImageNet",
):
    for normalization in (True, False):
        seed = 42
        # Noise augmentation:
        matching_options = {
            "representation": "core",
            "criterion": "mse",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
        }
        noise_type = {
            "add_noise": True,
            "noise_snr": None,
            "noise_std": {
                0.08: 0.1,
                0.12: 0.1,
                0.18: 0.1,
                0.26: 0.1,
                0.38: 0.1,
                -1: 0.5,
            },
        }

        experiments[
            Description(
                name="clean (normalization: {})".format(normalization), seed=seed
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=normalization,
                add_corrupted_test=True,
                download=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls, dataset_cls=dataset_cls
            ),
            trainer=trainer.TrainerConfig(
                comment="clean (normalization: {})".format(normalization),
                max_iter=90,
                lr_milestones=(30, 60),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                verbose=True,
            ),
            seed=seed,
        )

        experiments[
            Description(
                name="noisy (normalization: {})".format(normalization), seed=seed
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=normalization,
                add_corrupted_test=True,
                download=False,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls, dataset_cls=dataset_cls
            ),
            trainer=trainer.TrainerConfig(
                comment="noisy (normalization: {})".format(normalization),
                max_iter=90,
                lr_milestones=(30, 60),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                verbose=True,
                **noise_type
            ),
            seed=seed,
        )

        experiments[
            Description(
                name="noisy + rep matching (normalization: {})".format(normalization),
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=normalization,
                add_corrupted_test=True,
                download=False,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                representation_matching=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="noisy + rep. matching (normalization: {})".format(normalization),
                max_iter=90,
                lr_milestones=(30, 60),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                verbose=True,
                representation_matching=matching_options,
                **noise_type
            ),
            seed=seed,
        )

transfer_experiments = experiments
