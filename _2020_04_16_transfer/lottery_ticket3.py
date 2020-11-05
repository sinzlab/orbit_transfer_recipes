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
    for seed in (42, 8,
                 # 13, 23, 19
                 ):
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

        matching_options = {
            "representation": "core",
            "criterion": "mse",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
        }

        experiments[Description(name="CIFAR10: Clean", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18",
                conv_stem_kernel_size=3,
            ),
            trainer=trainer.TrainerConfig(
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                max_iter=80,
                lr_milestones=(50, 65),
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                comment="Clean normal SGD",
                verbose=True,
            ),
            seed=seed,
        )

        experiments[
            Description(name="CIFAR10: Noise Augmented", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18",
                conv_stem_kernel_size=3,
            ),
            trainer=trainer.TrainerConfig(
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                max_iter=80,
                lr_milestones=(50, 65),
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                comment="Noisy normal SGD",
                verbose=True,
                **noise_type,
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(name=dataset_cls + ": Clean", seed=seed)
        ] = TransferExperiment(
            [experiments[Description(name="CIFAR10: Clean", seed=seed)]]
        )
        transfer_experiments[
            Description(name=dataset_cls + ": Noisy", seed=seed)
        ] = TransferExperiment(
            [experiments[Description(name="CIFAR10: Noise Augmented", seed=seed)]]
        )

        for sparsity in (
            80,
            90,
            98,
        ):
            experiments[
                Description(
                    name="CIFAR10: Clean + Pruning {}".format(sparsity), seed=seed
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=80,
                    lr_milestones=(50, 65),
                    lr_decay=0.1,
                    lottery_ticket={
                        "rounds": 16,
                        "round_length": 80,
                        "percent_to_prune": sparsity,
                        "pruning": True,
                        "reinit": False,
                        "global_pruning": True,
                    },
                    adaptive_lr=False,
                    restore_best=False,
                    early_stop=False,
                    patience=1000,
                    comment="Clean lottery-ticket SGD",
                    verbose=True,
                ),
                seed=seed,
            )

            experiments[
                Description(
                    name="CIFAR10: Noise Augmented + Pruning {}".format(sparsity),
                    seed=seed,
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=80,
                    lr_milestones=(50, 65),
                    lr_decay=0.1,
                    lottery_ticket={
                        "rounds": 16,
                        "round_length": 80,
                        "percent_to_prune": sparsity,
                        "pruning": True,
                        "reinit": False,
                        "global_pruning": True,
                    },
                    adaptive_lr=False,
                    restore_best=False,
                    early_stop=False,
                    patience=1000,
                    comment="Noise Augmented lottery-ticket SGD",
                    verbose=True,
                    **noise_type,
                ),
                seed=seed,
            )

            experiments[
                Description(
                    name="CIFAR10: Noise Augmented + Repr Matching + Pruning {}".format(
                        sparsity
                    ),
                    seed=seed,
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                    representation_matching=True,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=80,
                    lr_milestones=(50, 65),
                    lr_decay=0.1,
                    lottery_ticket={
                        "rounds": 16,
                        "round_length": 80,
                        "percent_to_prune": sparsity,
                        "pruning": True,
                        "reinit": False,
                        "global_pruning": True,
                    },
                    adaptive_lr=False,
                    restore_best=False,
                    early_stop=False,
                    patience=1000,
                    comment="Noise Augmented lottery-ticket SGD",
                    verbose=True,
                    representation_matching=matching_options,
                    **noise_type,
                ),
                seed=seed,
            )

            experiments[
                Description(name="CIFAR10: Retrain {}".format(sparsity), seed=seed)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=80,
                    lr_milestones=(50, 65),
                    lr_decay=0.1,
                    adaptive_lr=False,
                    restore_best=False,
                    early_stop=False,
                    patience=1000,
                    lottery_ticket={
                        # "rounds": 4,
                        "percent_to_prune": sparsity,
                        "pruning": False,
                        "reinit": False,
                        "global_pruning": True,
                    },
                    comment="Retrain LT SGD",
                    verbose=True,
                ),
                seed=seed,
            )

            experiments[
                Description(
                    name="CIFAR10: Retrain {} (Noise Augmented)".format(sparsity),
                    seed=seed,
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=80,
                    lr_milestones=(50, 65),
                    lr_decay=0.1,
                    adaptive_lr=False,
                    restore_best=False,
                    early_stop=False,
                    patience=1000,
                    lottery_ticket={
                        # "rounds": 4,
                        "percent_to_prune": sparsity,
                        "pruning": False,
                        "reinit": False,
                        "global_pruning": True,
                    },
                    comment="Retrain LT SGD + Noise",
                    verbose=True,
                    **noise_type,
                ),
                seed=seed,
            )

            transfer_experiments[
                Description(
                    name=dataset_cls + ": Clean (LT {}) -> Retrain".format(sparsity),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name="CIFAR10: Clean + Pruning {}".format(sparsity),
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name="CIFAR10: Retrain {}".format(sparsity), seed=seed
                        )
                    ],
                ]
            )
            transfer_experiments[
                Description(
                    name=dataset_cls
                    + ": Noise Augmented (LT {}) -> Retrain".format(sparsity),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name="CIFAR10: Noise Augmented + Pruning {}".format(
                                sparsity
                            ),
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name="CIFAR10: Retrain {}".format(sparsity), seed=seed
                        )
                    ],
                ]
            )
            transfer_experiments[
                Description(
                    name=dataset_cls
                    + ": Clean (LT {}) -> Retrain (Noise Augmented)".format(sparsity),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name="CIFAR10: Clean + Pruning {}".format(sparsity),
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name="CIFAR10: Retrain {} (Noise Augmented)".format(
                                sparsity
                            ),
                            seed=seed,
                        )
                    ],
                ]
            )
            transfer_experiments[
                Description(
                    name=dataset_cls
                    + ": Noise Augmented (LT {}) -> Retrain (Noise Augmented)".format(
                        sparsity
                    ),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name="CIFAR10: Noise Augmented + Pruning {}".format(
                                sparsity
                            ),
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name="CIFAR10: Retrain {} (Noise Augmented)".format(
                                sparsity
                            ),
                            seed=seed,
                        )
                    ],
                ]
            )
