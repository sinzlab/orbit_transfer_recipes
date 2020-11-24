from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer


experiments = {}
transfer_experiments = {}


for dataset_cls in (
    ("CIFAR10", "CIFAR10"),
    # ("CIFAR100", "CIFAR10"),
    # ("SVHN", "MNIST"),
    # "TinyImageNet",
):
    for seed in (
        42,
        # 23,
        # 8
    ):
        transfer_no_freeze = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                apply_data_normalization=True,
                add_corrupted_test=dataset_cls[1] not in ("MNIST", "SVHN"),
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                type="resnet18"
                if dataset_cls[1] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls[1] == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Transfer",
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                freeze_bn=False,
                eval_with_bn_train=True,
                reset_linear=False,
                max_iter=90 if dataset_cls[1] == "TinyImageNet" else 200,
                scheduler="manual",
                scheduler_options={"milestones": (60, 120,160)},
            ),
            seed=seed,
        )
        transfer_freeze = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                apply_data_normalization=True,
                add_corrupted_test=dataset_cls[1] not in ("MNIST", "SVHN"),
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                type="resnet18"
                if dataset_cls[1] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls[1] == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Transfer",
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                freeze=("core",),
                freeze_bn=True,
                eval_with_bn_train=True,
                reset_linear=True,
                max_iter=90 if dataset_cls[1] == "TinyImageNet" else 200,
                scheduler="manual",
                scheduler_options={"milestones": (60, 120,160)},
            ),
            seed=seed,
        )
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
        experiments[
            Description(name=dataset_cls[0] + ": Clean", seed=seed,)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls[0],
                dataset_cls=dataset_cls[0],
                apply_data_normalization=True,
                add_corrupted_test=dataset_cls[0] not in ("MNIST", "SVHN"),
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls[0],
                dataset_cls=dataset_cls[0],
                type="resnet18"
                if dataset_cls[0] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls[0] == "TinyImageNet" else 3,
                representation_matching=False,
            ),
            trainer=trainer.TrainerConfig(
                comment="SGD Clean",
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                lr_warmup=10,
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                verbose=True,
                eval_with_bn_train=True,
                max_iter=90 if dataset_cls[0] == "TinyImageNet" else 200,
                scheduler="manual",
                scheduler_options={"milestones": (60, 120,160)},
            ),
            seed=seed,
        )
        experiments[
            Description(name=dataset_cls[0] + ": Noise Augmented", seed=seed,)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls[0],
                dataset_cls=dataset_cls[0],
                apply_data_normalization=True,
                add_corrupted_test=dataset_cls[0] not in ("MNIST", "SVHN"),
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls[0],
                dataset_cls=dataset_cls[0],
                type="resnet18"
                if dataset_cls[0] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls[0] == "TinyImageNet" else 3,
                representation_matching=False,
            ),
            trainer=trainer.TrainerConfig(
                comment="SGD Noise Augmented",
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                lr_warmup=10,
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                verbose=True,
                eval_with_bn_train=True,
                max_iter=90 if dataset_cls[0] == "TinyImageNet" else 200,
                scheduler="manual",
                scheduler_options={"milestones": (60, 120,160)},
                **noise_type
            ),
            seed=seed,
        )

        experiments[
            Description(name=dataset_cls[1] + ": Noise Augmented", seed=seed,)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                apply_data_normalization=True,
                add_corrupted_test=dataset_cls[1] not in ("MNIST", "SVHN"),
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                type="resnet18"
                if dataset_cls[1] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls[1] == "TinyImageNet" else 3,
                representation_matching=False,
            ),
            trainer=trainer.TrainerConfig(
                comment="SGD Noise Augmented",
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                lr_warmup=10,
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                verbose=True,
                eval_with_bn_train=True,
                max_iter=90 if dataset_cls[1] == "TinyImageNet" else 200,
                scheduler="manual",
                scheduler_options={"milestones": (60, 120,160)},
                **noise_type
            ),
            seed=seed,
        )
        experiments[
            Description(name=dataset_cls[1] + ": Clean", seed=seed,)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                apply_data_normalization=True,
                add_corrupted_test=dataset_cls[0] not in ("MNIST", "SVHN"),
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                type="resnet18"
                if dataset_cls[1] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls[1] == "TinyImageNet" else 3,
                representation_matching=False,
            ),
            trainer=trainer.TrainerConfig(
                comment="SGD Clean",
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                lr_warmup=10,
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                verbose=True,
                eval_with_bn_train=True,
                max_iter=90 if dataset_cls[1] == "TinyImageNet" else 200,
                scheduler="manual",
                scheduler_options={"milestones": (60, 120,160)},
            ),
            seed=seed,
        )


        transfer_experiments[
            Description(name="{} Clean".format(dataset_cls[1]), seed=seed,)
        ] = TransferExperiment(
            [experiments[Description(name=dataset_cls[1] + ": Clean", seed=seed,)],]
        )
        transfer_experiments[
            Description(name="{} Noise Augmented".format(dataset_cls[1]), seed=seed,)
        ] = TransferExperiment(
            [
                experiments[
                    Description(name=dataset_cls[1] + ": Noise Augmented", seed=seed,)
                ],
            ]
        )
        # transfer_experiments[
        #     Description(
        #         name="{} Clean -> {} Transfer (No Freeze)".format(
        #             dataset_cls[0], dataset_cls[1]
        #         ),
        #         seed=seed,
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[Description(name=dataset_cls[0] + ": Clean", seed=seed,)],
        #         transfer_no_freeze,
        #     ]
        # )
        # transfer_experiments[
        #     Description(
        #         name="{} Noise Augmented -> {} Transfer (No freeze)".format(
        #             dataset_cls[0], dataset_cls[1]
        #         ),
        #         seed=seed,
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(name=dataset_cls[0] + ": Noise Augmented", seed=seed,)
        #         ],
        #         transfer_no_freeze,
        #     ]
        # )
        # transfer_experiments[
        #     Description(
        #         name="{} Clean -> {} Transfer (Core + BN freeze)".format(
        #             dataset_cls[0], dataset_cls[1]
        #         ),
        #         seed=seed,
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[Description(name=dataset_cls[0] + ": Clean", seed=seed,)],
        #         transfer_freeze,
        #     ]
        # )
        # transfer_experiments[
        #     Description(
        #         name="{} Noise Augmented -> {} Transfer (Core + BN freeze)".format(
        #             dataset_cls[0], dataset_cls[1]
        #         ),
        #         seed=seed,
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(name=dataset_cls[0] + ": Noise Augmented", seed=seed,)
        #         ],
        #         transfer_freeze,
        #     ]
        # )

        for sparsity in (
            80,
            90,
            98,
        ):
            transfer_lt = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls[1],
                    dataset_cls=dataset_cls[1],
                    apply_data_normalization=True,
                    add_corrupted_test=dataset_cls[1] not in ("MNIST", "SVHN"),
                    download=True,
                    batch_size=256,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls[1],
                    dataset_cls=dataset_cls[1],
                    type="resnet18"
                    if dataset_cls[1] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                    else "resnet50",
                    conv_stem_kernel_size=5 if dataset_cls[1] == "TinyImageNet" else 3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=80,
                    scheduler="manual",
                    scheduler_options={"milestones":(50, 65)},
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
                    freeze_bn=False,
                    eval_with_bn_train=True,
                    reset_linear=False,
                ),
                seed=seed,
            )
            transfer_lt_no_train = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls[1],
                    dataset_cls=dataset_cls[1],
                    apply_data_normalization=True,
                    add_corrupted_test=dataset_cls[1] not in ("MNIST", "SVHN"),
                    download=True,
                    batch_size=256,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls[1],
                    dataset_cls=dataset_cls[1],
                    type="resnet18"
                    if dataset_cls[1] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                    else "resnet50",
                    conv_stem_kernel_size=5 if dataset_cls[1] == "TinyImageNet" else 3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=0,
                    scheduler="manual",
                    scheduler_options={"milestones":(50, 65)},
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
                    freeze_bn=False,
                    eval_with_bn_train=True,
                    reset_linear=False,
                ),
                seed=seed,
            )
            transfer_lt_noisy = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls[1],
                    dataset_cls=dataset_cls[1],
                    apply_data_normalization=True,
                    add_corrupted_test=dataset_cls[1] not in ("MNIST", "SVHN"),
                    download=True,
                    batch_size=256,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls[1],
                    dataset_cls=dataset_cls[1],
                    type="resnet18"
                    if dataset_cls[1] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                    else "resnet50",
                    conv_stem_kernel_size=5 if dataset_cls[1] == "TinyImageNet" else 3,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=0,
                    scheduler="manual",
                    scheduler_options={"milestones":(50, 65)},
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
                    freeze_bn=False,
                    eval_with_bn_train=True,
                    reset_linear=False,
                    **noise_type
                ),
                seed=seed,
            )
            experiments[
                Description(name=dataset_cls[0] + ": Clean Pruning {}".format(sparsity), seed=seed,)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls[0],
                    dataset_cls=dataset_cls[0],
                    apply_data_normalization=True,
                    add_corrupted_test=dataset_cls[0] not in ("MNIST", "SVHN"),
                    download=True,
                    batch_size=256,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls[0],
                    dataset_cls=dataset_cls[0],
                    type="resnet18"
                    if dataset_cls[0] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                    else "resnet50",
                    conv_stem_kernel_size=5 if dataset_cls[0] == "TinyImageNet" else 3,
                    representation_matching=False,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=80,
                    scheduler="manual",
                    scheduler_options={"milestones":(50, 65)},
                    lr_warmup=10,
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
                    eval_with_bn_train=True,
                ),
                seed=seed,
            )


            experiments[
                Description(name=dataset_cls[0] + ": Noise Augmented Pruning {}".format(sparsity), seed=seed,)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=dataset_cls[0],
                    dataset_cls=dataset_cls[0],
                    apply_data_normalization=True,
                    add_corrupted_test=dataset_cls[0] not in ("MNIST", "SVHN"),
                    download=True,
                    batch_size=256,
                ),
                model=model.ClassificationModelConfig(
                    comment=dataset_cls[0],
                    dataset_cls=dataset_cls[0],
                    type="resnet18"
                    if dataset_cls[0] in ("CIFAR10", "CIFAR100", "MNIST", "SVHN")
                    else "resnet50",
                    conv_stem_kernel_size=5 if dataset_cls[0] == "TinyImageNet" else 3,
                    representation_matching=False,
                ),
                trainer=trainer.TrainerConfig(
                    optimizer="SGD",
                    optimizer_options={"lr": 0.01, "momentum": 0.9},
                    max_iter=80,
                    scheduler="manual",
                    scheduler_options={"milestones":(50, 65)},
                    lr_warmup=10,
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
                    eval_with_bn_train=True,
                    **noise_type
                ),
                seed=seed,
            )
            # transfer_experiments[
            #     Description(
            #         name="{} Clean Pruning -> {} Transfer (LT, {}, No Train)".format(
            #             dataset_cls[0], dataset_cls[1], sparsity
            #         ),
            #         seed=seed,
            #     )
            # ] = TransferExperiment(
            #     [
            #         experiments[
            #             Description(name=dataset_cls[0] + ": Clean Pruning {}".format(sparsity), seed=seed,)
            #         ],
            #         transfer_lt_no_train,
            #     ]
            # )
            # transfer_experiments[
            #     Description(
            #         name="{} Noise Augmented Pruning -> {} Transfer (LT, {}, No Train)".format(
            #             dataset_cls[0], dataset_cls[1], sparsity
            #         ),
            #         seed=seed,
            #     )
            # ] = TransferExperiment(
            #     [
            #         experiments[
            #             Description(
            #                 name=dataset_cls[0] + ": Noise Augmented Pruning {}".format(sparsity), seed=seed,
            #             )
            #         ],
            #         transfer_lt_no_train,
            #     ]
            # )
            # transfer_experiments[
            #     Description(
            #         name="{} Clean Pruning -> {} Transfer (LT, {})".format(
            #             dataset_cls[0], dataset_cls[1], sparsity
            #         ),
            #         seed=seed,
            #     )
            # ] = TransferExperiment(
            #     [
            #         experiments[
            #             Description(name=dataset_cls[0] + ": Clean Pruning {}".format(sparsity), seed=seed,)
            #         ],
            #         transfer_lt,
            #     ]
            # )
            transfer_experiments[
                Description(
                    name="{} Noise Augmented Pruning -> {} Transfer (LT, {})".format(
                        dataset_cls[0], dataset_cls[1], sparsity
                    ),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name=dataset_cls[0] + ": Noise Augmented Pruning {}".format(sparsity), seed=seed,
                        )
                    ],
                    transfer_lt,
                ]
            )
            transfer_experiments[
                Description(
                    name="{} Clean Pruning -> {} Noise Augmented Transfer (LT, {})".format(
                        dataset_cls[0], dataset_cls[1], sparsity
                    ),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(name=dataset_cls[0] + ": Clean Pruning {}".format(sparsity), seed=seed,)
                    ],
                    transfer_lt_noisy,
                ]
            )
            transfer_experiments[
                Description(
                    name="{} Noise Augmented Pruning -> {} Noise Augmented Transfer (LT, {})".format(
                        dataset_cls[0], dataset_cls[1], sparsity
                    ),
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name=dataset_cls[0] + ": Noise Augmented Pruning {}".format(sparsity), seed=seed,
                        )
                    ],
                    transfer_lt_noisy,
                ]
            )
