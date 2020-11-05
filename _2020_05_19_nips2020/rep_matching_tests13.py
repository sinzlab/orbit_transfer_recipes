from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

for dataset_cls in (
    # "CIFAR100",
    "CIFAR10",
    # "TinyImageNet",
):
    for seed in (42,
                 # 23,
                 # 8
                 ):
        # Noise augmentation:
        noise_type = {
            "add_noise": True,
            "noise_snr": None,
            "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5},
        }
        noise_adv = {"noise_adv_classification": False, "noise_adv_regression": True}

        matching_options = {
            "representation": "conv_rep",
            "criterion": "mse",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
        }
        experiments[
            Description(
                name=dataset_cls + ": Noise Augmented + Repr Matching (Euclid; clean only; 13)", seed=seed
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=128,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type=18 if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching (Euclid; clean only; 13)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60) if dataset_cls == "TinyImageNet" else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                representation_matching=matching_options,
                **noise_type
            ),
            seed=seed,
        )


        transfer = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=128,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type=18 if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Transfer + Reset",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60) if dataset_cls == "TinyImageNet" else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                freeze=("core",),
                reset_linear=True,
                patience=1000,
            ),
            seed=seed,
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (Euclid; clean only; 13) -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented + Repr Matching (Euclid; clean only; 13)",
                        seed=seed,
                    )
                ],
                transfer,
            ]
        )
