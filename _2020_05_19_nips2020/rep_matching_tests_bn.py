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
        # 23,
        # 8
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
        noise_adv = {"noise_adv_classification": False, "noise_adv_regression": True}

        matching_options = {
            "representations": ["core"],
            "criterion": "mse",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
        }

        experiments[
            Description(name=dataset_cls + ": Clean (fixed)", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented (fixed)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
            ),
            seed=seed,
        )

        experiments[
            Description(name=dataset_cls + ": Noise Augmented (fixed)", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented (fixed)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                **noise_type
            ),
            seed=seed,
        )

        experiments[
            Description(
                name=dataset_cls
                + ": Noise Augmented + Repr Matching (Euclid; clean only; fixed)",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching (Euclid; clean only; fixed)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                representation_matching=matching_options,
                **noise_type
            ),
            seed=seed,
        )

        matching_options = {
            "representations": ["readout"],
            "criterion": "mse",
            "combine_losses": "avg",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
        }
        experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (Logits)",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
                get_intermediate_rep={
                    "fc": "readout",
                },
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching (Euclid; clean only; fixed)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                representation_matching=matching_options,
                **noise_type
            ),
            seed=seed,
        )

        intermediate_outputs = {
            "conv1": "layer0.conv1",
            # layer1
            "layer1.0.conv1": "layer1.0.conv1",
            "layer1.0.conv2": "layer1.0.conv2",
            "layer1.1.conv1": "layer1.1.conv1",
            "layer1.1.conv2": "layer1.1.conv2",
            # layer2
            "layer2.0.conv1": "layer2.0.conv1",
            "layer2.0.conv2": "layer2.0.conv2",
            "layer2.1.conv1": "layer2.1.conv1",
            "layer2.1.conv2": "layer2.1.conv2",
            # layer3
            "layer3.0.conv1": "layer3.0.conv1",
            "layer3.0.conv2": "layer3.0.conv2",
            "layer3.1.conv1": "layer3.1.conv1",
            "layer3.1.conv2": "layer3.1.conv2",
            # layer4
            "layer4.0.conv1": "layer4.0.conv1",
            "layer4.0.conv2": "layer4.0.conv2",
            "layer4.1.conv1": "layer4.1.conv1",
            "layer4.1.conv2": "layer4.1.conv2",
            # core output
            "flatten": "core",
            "fc": "readout",
        }
        matching_options = {
            "representations": list(intermediate_outputs.values()),
            "criterion": "mse",
            "combine_losses": "avg",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
        }
        experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (All)",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
                get_intermediate_rep=intermediate_outputs,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching (Euclid; clean only; fixed)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                representation_matching=matching_options,
                **noise_type
            ),
            seed=seed,
        )

        intermediate_outputs = {
            "conv1": "layer0.conv1",
            # layer1
            "layer1.0.conv1": "layer1.0.conv1",
            "layer1.0.conv2": "layer1.0.conv2",
            "layer1.1.conv1": "layer1.1.conv1",
            "layer1.1.conv2": "layer1.1.conv2",
            # layer2
            "layer2.0.conv1": "layer2.0.conv1",
            "layer2.0.conv2": "layer2.0.conv2",
            "layer2.1.conv1": "layer2.1.conv1",
            "layer2.1.conv2": "layer2.1.conv2",
            # layer3
            "layer3.0.conv1": "layer3.0.conv1",
            "layer3.0.conv2": "layer3.0.conv2",
            "layer3.1.conv1": "layer3.1.conv1",
            "layer3.1.conv2": "layer3.1.conv2",
            # layer4
            "layer4.0.conv1": "layer4.0.conv1",
            "layer4.0.conv2": "layer4.0.conv2",
            "layer4.1.conv1": "layer4.1.conv1",
            "layer4.1.conv2": "layer4.1.conv2",
            # core output
        }
        matching_options = {
            "representations": list(intermediate_outputs.values()),
            "criterion": "mse",
            "combine_losses": "avg",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
        }
        experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (All in Core)",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
                get_intermediate_rep=intermediate_outputs,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching (Euclid; clean only; fixed)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                representation_matching=matching_options,
                **noise_type
            ),
            seed=seed,
        )

        intermediate_outputs = {
            "conv1": "layer0.conv1",
            # layer1
            "layer1.1.conv2": "layer1.1.conv2",
            # layer2
            "layer2.1.conv2": "layer2.1.conv2",
            # layer3
            "layer3.1.conv2": "layer3.1.conv2",
            # layer4
            "layer4.1.conv2": "layer4.1.conv2",
            # core output
        }
        matching_options = {
            "representations": list(intermediate_outputs.values()),
            "criterion": "mse",
            "combine_losses": "avg",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
        }
        experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (Block out)",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
                get_intermediate_rep=intermediate_outputs,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching (Euclid; clean only; fixed)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                representation_matching=matching_options,
                **noise_type
            ),
            seed=seed,
        )
        experiments[
            Description(
                name=dataset_cls + ": Noise Augmented + Adv Regression (fixed)",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                noise_adv_regression=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Adv Regression (fixed)",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                **noise_adv,
                **noise_type
            ),
            seed=seed,
        )

        transfer = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Transfer + Reset",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                freeze=("core",),
                freeze_bn=True,
                reset_linear=True,
                patience=1000,
            ),
            seed=seed,
        )

        transfer_normal = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                type="resnet18" if dataset_cls == "CIFAR10" else "resnet50",
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Transfer + Reset",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                freeze=("core",),
                freeze_bn=False,
                reset_linear=True,
                patience=1000,
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(name=dataset_cls + ": Clean (fixed)", seed=seed,)
        ] = TransferExperiment(
            [
                experiments[
                    Description(name=dataset_cls + ": Clean (fixed)", seed=seed,)
                ],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls + ": Noise Augmented (fixed) -> Transfer (freeze bn)", seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented (fixed)", seed=seed,
                    )
                ],
                transfer,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                + ": Noise Augmented + Repr Matching (Euclid; clean only; fixed) -> Transfer (freeze bn)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                        + ": Noise Augmented + Repr Matching (Euclid; clean only; fixed)",
                        seed=seed,
                    )
                ],
                transfer,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                + ": Noise Augmented + Adv Regression (fixed) -> Transfer (freeze bn)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented + Adv Regression (fixed)",
                        seed=seed,
                    )
                ],
                transfer,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (All) -> Transfer (freeze bn)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (All)",
                        seed=seed,
                    )
                ],
                transfer,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (Logits) -> Transfer (freeze bn)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (Logits)",
                        seed=seed,
                    )
                ],
                transfer,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (All in Core) -> Transfer (freeze bn)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (All in Core)",
                        seed=seed,
                    )
                ],
                transfer,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (Block out) -> Transfer (freeze bn)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (Block out)",
                        seed=seed,
                    )
                ],
                transfer,
            ]
        )

        transfer_experiments[
            Description(
                name=dataset_cls + ": Noise Augmented (fixed) -> Transfer", seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented (fixed)", seed=seed,
                    )
                ],
                transfer_normal,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (Euclid; clean only; fixed) -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (Euclid; clean only; fixed)",
                        seed=seed,
                    )
                ],
                transfer_normal,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Adv Regression (fixed) -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented + Adv Regression (fixed)",
                        seed=seed,
                    )
                ],
                transfer_normal,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (All) -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (All)",
                        seed=seed,
                    )
                ],
                transfer_normal,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (Logits) -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (Logits)",
                        seed=seed,
                    )
                ],
                transfer_normal,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (All in Core) -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (All in Core)",
                        seed=seed,
                    )
                ],
                transfer_normal,
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                     + ": Noise Augmented + Repr Matching (Block out) -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                             + ": Noise Augmented + Repr Matching (Block out)",
                        seed=seed,
                    )
                ],
                transfer_normal,
            ]
        )
