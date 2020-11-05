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
    for seed in (42, 8, 13):
        # Noise augmentation:
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

        experiments[Description(name="CIFAR10: Clean", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type=18,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                comment="Clean normal",
                verbose=True,
            ),
            seed=seed,
        )

        experiments[Description(name="CIFAR10: Noise Augmented", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type=18,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                comment="Noisy normal",
                verbose=True,
                **noise_type,
            ),
            seed=seed,
        )

        experiments[Description(name="CIFAR10: Clean + Pruning", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type=18,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Clean lottery-ticket",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=None if dataset_cls == "TinyImageNet" else None,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                lottery_ticket={
                    "rounds": 4,
                    "round_length": 50,
                    "percent_to_prune": 80,
                    "pruning": True,
                    "reinit": False,
                    "global_pruning": True,
                },
                verbose=True,
            ),
            seed=seed,
        )

        experiments[Description(name="CIFAR10: Noise Augmented + Pruning", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type=18,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented lottery-ticket",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=None if dataset_cls == "TinyImageNet" else None,
                verbose=True,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                lottery_ticket={
                    "rounds": 4,
                    "round_length": 50,
                    "percent_to_prune": 80,
                    "pruning": True,
                    "reinit": False,
                    "global_pruning": True,
                },
                **noise_type,
            ),
            seed=seed,
        )

        matching_options = {
            "representation": "conv_rep",
            "criterion": "mse",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
        }

        experiments[Description(name="CIFAR10: Noise Augmented + Rep Matching + Pruning", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type=18,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented lottery-ticket",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=None if dataset_cls == "TinyImageNet" else None,
                verbose=True,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                lottery_ticket={
                    "rounds": 4,
                    "round_length": 50,
                    "percent_to_prune": 80,
                    "pruning": True,
                    "reinit": False,
                    "global_pruning": True,
                },
                representation_matching=matching_options,
                **noise_type,
            ),
            seed=seed,
        )

        experiments[Description(name="transfer", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type=18,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                comment="Retrain LT",
                lottery_ticket={
                    # "rounds": 4,
                    "percent_to_prune": 80,
                    "pruning": False,
                    "reinit": False,
                    "global_pruning": True,
                },
                verbose=True,
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(name=dataset_cls + ": Clean", seed=seed)
        ] = TransferExperiment([experiments[Description(name="CIFAR10: Clean", seed=seed)]])
        transfer_experiments[
            Description(name=dataset_cls + ": Noisy", seed=seed)
        ] = TransferExperiment([experiments[Description(name="CIFAR10: Noise Augmented", seed=seed)]])
        transfer_experiments[
            Description(name=dataset_cls + ": Clean (LT) -> Retrain", seed=seed)
        ] = TransferExperiment(
            [
                experiments[Description(name="CIFAR10: Clean + Pruning", seed=seed)],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls + ": Noise Augmented (LT) -> Retrain", seed=seed
            )
        ] = TransferExperiment(
            [
                experiments[Description(name="CIFAR10: Noise Augmented + Pruning", seed=seed)],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )
