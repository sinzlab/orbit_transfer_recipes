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
    for seed in (
        42,
        8,
        # 13,
        23
        # , 19
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
        experiments[
            Description(name=dataset_cls + ": Clean (Amsgrad, Warmup)", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
                download=True,
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
                # lr_milestones=(50, 65),
                adaptive_lr=True,
                restore_best=True,
                early_stop=True,
                patience=1000,
                comment="Clean Amsgrad",
                verbose=True,
            ),
            seed=seed,
        )
        experiments[
            Description(name=dataset_cls + ": Noise Augmented (Amsgrad, Warmup)", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
                download=True,
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
                # lr_milestones=(50, 65),
                adaptive_lr=True,
                restore_best=True,
                early_stop=True,
                patience=1000,
                comment="Noisy Amsgrad",
                verbose=True,
                **noise_type,
            ),
            seed=seed,
        )

        experiments[
            Description(
                name=dataset_cls + ": Noise Augmented + Noise Adv (Amsgrad, Warmup)", seed=seed
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
                download=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18",
                conv_stem_kernel_size=3,
                noise_adv_regression=True,
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
                # lr_milestones=(50, 65),
                adaptive_lr=True,
                restore_best=True,
                early_stop=True,
                patience=1000,
                comment="Noisy Amsgrad + Noise Adv",
                verbose=True,
                noise_adv_regression=True,
                **noise_type,
            ),
            seed=seed,
        )

        experiments[
            Description(name=dataset_cls + ": Transfer (Amsgrad, Warmup)", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
                download=True,
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
                # lr_milestones=(50, 65),
                adaptive_lr=True,
                restore_best=True,
                early_stop=True,
                patience=1000,
                comment="Transfer",
                verbose=True,
                freeze=("core",),
                reset_linear=True,
            ),
            seed=seed,
        )

        experiments[
            Description(name=dataset_cls + ": Transfer (Amsgrad)", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
                download=True,
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
                # lr_warmup=10,
                max_iter=80,
                # lr_milestones=(50, 65),
                adaptive_lr=True,
                restore_best=True,
                early_stop=True,
                patience=1000,
                comment="Transfer",
                verbose=True,
                freeze=("core",),
                reset_linear=True,
            ),
            seed=seed,
        )


        experiments[
            Description(name=dataset_cls + ": Transfer (Warmup)", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=128,
                download=True,
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
                    "amsgrad": False,
                    "lr": 0.0003,
                    "weight_decay": 5e-4,
                },
                lr_decay=0.8,
                lr_warmup=10,
                max_iter=80,
                # lr_milestones=(50, 65),
                adaptive_lr=True,
                restore_best=True,
                early_stop=True,
                patience=1000,
                comment="Transfer",
                verbose=True,
                freeze=("core",),
                reset_linear=True,
            ),
            seed=seed,
        )

        # transfer_experiments[
        #     Description(
        #         name=dataset_cls
        #              + ": Noise Augmented + Noise Adv (Amsgrad, Warmup) -> Transfer (Amsgrad)",
        #         seed=seed,
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(
        #                 name=dataset_cls + ": Noise Augmented + Noise Adv (Amsgrad, Warmup)",
        #                 seed=seed,
        #             )
        #         ],
        #         experiments[
        #             Description(name=dataset_cls + ": Transfer (Amsgrad)", seed=seed)
        #         ],
        #     ]
        # )
        #
        # transfer_experiments[
        #     Description(name=dataset_cls + ": Clean (Amsgrad, Warmup) -> Transfer (Amsgrad)", seed=seed)
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(name=dataset_cls + ": Clean (Amsgrad, Warmup)", seed=seed)
        #         ],
        #         experiments[
        #             Description(name=dataset_cls + ": Transfer (Amsgrad)", seed=seed)
        #         ],
        #     ]
        # )
        #
        # transfer_experiments[
        #     Description(
        #         name=dataset_cls + ": Noise Augmented (Amsgrad, Warmup) -> Transfer (Amsgrad)", seed=seed
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(
        #                 name=dataset_cls + ": Noise Augmented (Amsgrad, Warmup)", seed=seed
        #             )
        #         ],
        #         experiments[
        #             Description(name=dataset_cls + ": Transfer (Amsgrad)", seed=seed)
        #         ],
        #     ]
        # )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Noise Adv (Amsgrad, Warmup) -> Transfer (Warmup)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented + Noise Adv (Amsgrad, Warmup)",
                        seed=seed,
                    )
                ],
                experiments[
                    Description(name=dataset_cls + ": Transfer (Warmup)", seed=seed)
                ],
            ]
        )

        # transfer_experiments[
        #     Description(name=dataset_cls + ": Clean (Amsgrad, Warmup) -> Transfer (Warmup)", seed=seed)
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(name=dataset_cls + ": Clean (Amsgrad, Warmup)", seed=seed)
        #         ],
        #         experiments[
        #             Description(name=dataset_cls + ": Transfer (Warmup)", seed=seed)
        #         ],
        #     ]
        # )

        transfer_experiments[
            Description(
                name=dataset_cls + ": Noise Augmented (Amsgrad, Warmup) -> Transfer (Warmup)", seed=seed
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented (Amsgrad, Warmup)", seed=seed
                    )
                ],
                experiments[
                    Description(name=dataset_cls + ": Transfer (Warmup)", seed=seed)
                ],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                + ": Noise Augmented + Noise Adv (Amsgrad, Warmup) -> Transfer (Amsgrad, Warmup)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented + Noise Adv (Amsgrad, Warmup)",
                        seed=seed,
                    )
                ],
                experiments[
                    Description(name=dataset_cls + ": Transfer (Amsgrad, Warmup)", seed=seed)
                ],
            ]
        )

        # transfer_experiments[
        #     Description(name=dataset_cls + ": Clean (Amsgrad, Warmup) -> Transfer (Amsgrad, Warmup)", seed=seed)
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(name=dataset_cls + ": Clean (Amsgrad, Warmup)", seed=seed)
        #         ],
        #         experiments[
        #             Description(name=dataset_cls + ": Transfer (Amsgrad, Warmup)", seed=seed)
        #         ],
        #     ]
        # )

        transfer_experiments[
            Description(
                name=dataset_cls + ": Noise Augmented (Amsgrad, Warmup) -> Transfer (Amsgrad, Warmup)", seed=seed
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented (Amsgrad, Warmup)", seed=seed
                    )
                ],
                experiments[
                    Description(name=dataset_cls + ": Transfer (Amsgrad, Warmup)", seed=seed)
                ],
            ]
        )


        # experiments[
        #     Description(name=dataset_cls + ": Clean", seed=seed)
        # ] = Experiment(
        #     dataset=dataset.ImageDatasetConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         apply_data_normalization=True,
        #         add_corrupted_test=True,
        #         batch_size=128,
        #         download=True,
        #     ),
        #     model=model.ClassificationModelConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         type="resnet18",
        #         conv_stem_kernel_size=3,
        #     ),
        #     trainer=trainer.TrainerConfig(
        #         optimizer="Adam",
        #         optimizer_options={
        #             "amsgrad": False,
        #             "lr": 0.0003,
        #             "weight_decay": 5e-4,
        #         },
        #         lr_decay=0.8,
        #         # lr_warmup=10,
        #         max_iter=80,
        #         lr_milestones=(50, 65),
        #         adaptive_lr=False,
        #         restore_best=False,
        #         early_stop=False,
        #         patience=1000,
        #         comment="Clean",
        #         verbose=True,
        #     ),
        #     seed=seed,
        # )
        # experiments[
        #     Description(name=dataset_cls + ": Noise Augmented", seed=seed)
        # ] = Experiment(
        #     dataset=dataset.ImageDatasetConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         apply_data_normalization=True,
        #         add_corrupted_test=True,
        #         batch_size=128,
        #         download=True,
        #     ),
        #     model=model.ClassificationModelConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         type="resnet18",
        #         conv_stem_kernel_size=3,
        #     ),
        #     trainer=trainer.TrainerConfig(
        #         optimizer="Adam",
        #         optimizer_options={
        #             "amsgrad": False,
        #             "lr": 0.0003,
        #             "weight_decay": 5e-4,
        #         },
        #         lr_decay=0.8,
        #         # lr_warmup=10,
        #         max_iter=80,
        #         lr_milestones=(50, 65),
        #         adaptive_lr=False,
        #         restore_best=False,
        #         early_stop=False,
        #         patience=1000,
        #         comment="Noisy",
        #         verbose=True,
        #         **noise_type,
        #     ),
        #     seed=seed,
        # )
        #
        # experiments[
        #     Description(
        #         name=dataset_cls + ": Noise Augmented + Noise Adv", seed=seed
        #     )
        # ] = Experiment(
        #     dataset=dataset.ImageDatasetConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         apply_data_normalization=True,
        #         add_corrupted_test=True,
        #         batch_size=128,
        #         download=True,
        #     ),
        #     model=model.ClassificationModelConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         type="resnet18",
        #         conv_stem_kernel_size=3,
        #         noise_adv_regression=True,
        #     ),
        #     trainer=trainer.TrainerConfig(
        #         optimizer="Adam",
        #         optimizer_options={
        #             "amsgrad": False,
        #             "lr": 0.0003,
        #             "weight_decay": 5e-4,
        #         },
        #         lr_decay=0.8,
        #         # lr_warmup=10,
        #         max_iter=80,
        #         lr_milestones=(50, 65),
        #         adaptive_lr=False,
        #         restore_best=False,
        #         early_stop=False,
        #         patience=1000,
        #         comment="Noisy + Noise Adv",
        #         verbose=True,
        #         noise_adv_regression=True,
        #         **noise_type,
        #     ),
        #     seed=seed,
        # )
        #
        # experiments[
        #     Description(name=dataset_cls + ": Transfer", seed=seed)
        # ] = Experiment(
        #     dataset=dataset.ImageDatasetConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         apply_data_normalization=True,
        #         add_corrupted_test=True,
        #         batch_size=128,
        #         download=True,
        #     ),
        #     model=model.ClassificationModelConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         type="resnet18",
        #         conv_stem_kernel_size=3,
        #     ),
        #     trainer=trainer.TrainerConfig(
        #         optimizer="Adam",
        #         optimizer_options={
        #             "amsgrad": False,
        #             "lr": 0.0003,
        #             "weight_decay": 5e-4,
        #         },
        #         lr_decay=0.8,
        #         # lr_warmup=10,
        #         max_iter=80,
        #         lr_milestones=(50, 65),
        #         adaptive_lr=False,
        #         restore_best=False,
        #         early_stop=False,
        #         patience=1000,
        #         comment="Transfer",
        #         verbose=True,
        #         freeze=("core",),
        #         reset_linear=True,
        #     ),
        #     seed=seed,
        # )

        # transfer_experiments[
        #     Description(
        #         name=dataset_cls
        #              + ": Noise Augmented + Noise Adv -> Transfer",
        #         seed=seed,
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(
        #                 name=dataset_cls + ": Noise Augmented + Noise Adv",
        #                 seed=seed,
        #             )
        #         ],
        #         experiments[
        #             Description(name=dataset_cls + ": Transfer", seed=seed)
        #         ],
        #     ]
        # )
        #
        # transfer_experiments[
        #     Description(name=dataset_cls + ": Clean -> Transfer", seed=seed)
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(name=dataset_cls + ": Clean", seed=seed)
        #         ],
        #         experiments[
        #             Description(name=dataset_cls + ": Transfer", seed=seed)
        #         ],
        #     ]
        # )
        #
        # transfer_experiments[
        #     Description(
        #         name=dataset_cls + ": Noise Augmented -> Transfer", seed=seed
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[
        #             Description(
        #                 name=dataset_cls + ": Noise Augmented", seed=seed
        #             )
        #         ],
        #         experiments[
        #             Description(name=dataset_cls + ": Transfer", seed=seed)
        #         ],
        #     ]
        # )
