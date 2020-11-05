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

        experiments[Description(name="CIFAR10: Clean (Adam)", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
                download=True
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18",
                conv_stem_kernel_size=3,
            ),
            trainer=trainer.TrainerConfig(
                optimizer="Adam",
                optimizer_options={
                    "amsgrad": True,
                    "lr": 0.0003,
                    "weight_decay": 5e-4,
                },
                lr_decay=0.8,
                lr_warmup=10,
                max_iter=80,
                lr_milestones=(50, 65),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                comment="Clean normal Adam",
                verbose=True,
            ),
            seed=seed,
        )

        experiments[
            Description(name="CIFAR10: Noise Augmented (Adam)", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
                download=True
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18",
                conv_stem_kernel_size=3,
            ),
            trainer=trainer.TrainerConfig(
                optimizer="Adam",
                optimizer_options={
                    "amsgrad": True,
                    "lr": 0.0003,
                    "weight_decay": 5e-4,
                },
                lr_decay=0.8,
                lr_warmup=10,
                max_iter=80,
                lr_milestones=(50, 65),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                comment="Noisy normal Adam",
                verbose=True,
                **noise_type,
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(name=dataset_cls + ": Clean (Adam)", seed=seed)
        ] = TransferExperiment(
            [experiments[Description(name="CIFAR10: Clean (Adam)", seed=seed)]]
        )
        transfer_experiments[
            Description(name=dataset_cls + ": Noisy (Adam)", seed=seed)
        ] = TransferExperiment(
            [experiments[Description(name="CIFAR10: Noise Augmented (Adam)", seed=seed)]]
        )

        for sparsity in (
                80,
                90,
                98,
        ):
            experiments[
                Description(
                    name="CIFAR10: Clean + Pruning {} (Adam)".format(sparsity), seed=seed
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                    download=True
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="Adam",
                    optimizer_options={
                        "amsgrad": True,
                        "lr": 0.0003,
                        "weight_decay": 5e-4,
                    },
                    lr_decay=0.8,
                    lr_warmup=10,
                    max_iter=80,
                    lr_milestones=(50, 65),
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
                    comment="Clean lottery-ticket Adam",
                    verbose=True,
                ),
                seed=seed,
            )

            experiments[
                Description(
                    name="CIFAR10: Noise Augmented + Pruning {} (Adam)".format(sparsity),
                    seed=seed,
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                    download=True
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="Adam",
                    optimizer_options={
                        "amsgrad": True,
                        "lr": 0.0003,
                        "weight_decay": 5e-4,
                    },
                    lr_decay=0.8,
                    lr_warmup=10,
                    max_iter=80,
                    lr_milestones=(50, 65),
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
                    comment="Noise Augmented lottery-ticket Adam",
                    verbose=True,
                    **noise_type,
                ),
                seed=seed,
            )

            experiments[
                Description(
                    name="CIFAR10: Noise Augmented + Repr Matching + Pruning {} (Adam)".format(
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
                    download=True
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                    representation_matching=True,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="Adam",
                    optimizer_options={
                        "amsgrad": True,
                        "lr": 0.0003,
                        "weight_decay": 5e-4,
                    },
                    lr_decay=0.8,
                    lr_warmup=10,
                    max_iter=80,
                    lr_milestones=(50, 65),
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
                    comment="Noise Augmented lottery-ticket Adam",
                    verbose=True,
                    representation_matching=matching_options,
                    **noise_type,
                ),
                seed=seed,
            )

            experiments[
                Description(name="CIFAR10: Retrain {} (Adam)".format(sparsity), seed=seed)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                    download=True
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="Adam",
                    optimizer_options={
                        "amsgrad": True,
                        "lr": 0.0003,
                        "weight_decay": 5e-4,
                    },
                    lr_decay=0.8,
                    lr_warmup=10,
                    max_iter=80,
                    lr_milestones=(50, 65),
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
                    comment="Retrain LT Adam",
                    verbose=True,
                ),
                seed=seed,
            )

            experiments[
                Description(
                    name="CIFAR10: Retrain {} (Noise Augmented) (Adam)".format(sparsity),
                    seed=seed,
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    apply_data_normalization=True,
                    add_corrupted_test=True,
                    batch_size=128,
                    download=True
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls,
                    dataset_cls=dataset_cls,
                    type="resnet18",
                    conv_stem_kernel_size=3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="Adam",
                    optimizer_options={
                        "amsgrad": True,
                        "lr": 0.0003,
                        "weight_decay": 5e-4,
                    },
                    lr_decay=0.8,
                    lr_warmup=10,
                    max_iter=80,
                    lr_milestones=(50, 65),
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
                    comment="Retrain LT Adam + Noise",
                    verbose=True,
                    **noise_type,
                ),
                seed=seed,
            )

            transfer_experiments[
                Description(
                    name=dataset_cls + ": Clean (LT {}) -> Retrain (Adam)".format(sparsity),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name="CIFAR10: Clean + Pruning {} (Adam)".format(sparsity),
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name="CIFAR10: Retrain {} (Adam)".format(sparsity), seed=seed
                        )
                    ],
                ]
            )
            transfer_experiments[
                Description(
                    name=dataset_cls
                         + ": Noise Augmented (LT {}) -> Retrain (Adam)".format(sparsity),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name="CIFAR10: Noise Augmented + Pruning {} (Adam)".format(
                                sparsity
                            ),
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name="CIFAR10: Retrain {} (Adam)".format(sparsity), seed=seed
                        )
                    ],
                ]
            )
            transfer_experiments[
                Description(
                    name=dataset_cls
                         + ": Clean (LT {}) -> Retrain (Noise Augmented) (Adam)".format(sparsity),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name="CIFAR10: Clean + Pruning {} (Adam)".format(sparsity),
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name="CIFAR10: Retrain {} (Noise Augmented) (Adam)".format(
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
                         + ": Noise Augmented (LT {}) -> Retrain (Noise Augmented) (Adam)".format(
                        sparsity
                    ),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name="CIFAR10: Noise Augmented + Pruning {} (Adam)".format(
                                sparsity
                            ),
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name="CIFAR10: Retrain {} (Noise Augmented) (Adam)".format(
                                sparsity
                            ),
                            seed=seed,
                        )
                    ],
                ]
            )
