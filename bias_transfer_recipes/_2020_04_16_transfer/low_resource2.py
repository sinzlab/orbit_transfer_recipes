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
):
    for seed in (42,):
        # Noise augmentation:
        matching_options = {
            "representation": "core",
            "criterion": "mse",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
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
        dataset_large = dataset.ImageDatasetConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            apply_data_normalization=True,
            add_corrupted_test=True,
            filter_classes=(0, 180),
            download=True,
        )
        dataset_small = dataset.ImageDatasetConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            apply_data_normalization=True,
            add_corrupted_test=True,
            filter_classes=(180, 200),
            download=True,
        )

        experiments[
            Description(name=dataset_cls + " 90% : Clean", seed=seed)
        ] = Experiment(
            dataset=dataset_large,
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=False,
                num_classes=180,
            ),
            trainer=trainer.TrainerConfig(
                comment="Clean Training",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                verbose=True,
                loss_functions={"img_classification": "CrossEntropyLoss"},
            ),
            seed=seed,
        )

        experiments[
            Description(name=dataset_cls + " 90% : Noise Augmented", seed=seed)
        ] = Experiment(
            dataset=dataset_large,
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=False,
                num_classes=180,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching Training",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                verbose=True,
                loss_functions={"img_classification": "CrossEntropyLoss"},
                **noise_type,
            ),
            seed=seed,
        )

        experiments[
            Description(
                name=dataset_cls + " 90% : Noise Augmented + Repr. Matching", seed=seed
            )
        ] = Experiment(
            dataset=dataset_large,
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
                num_classes=180,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching Training",
                representation_matching=matching_options,
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                verbose=True,
                loss_functions={"img_classification": "CrossEntropyLoss"},
                **noise_type,
            ),
            seed=seed,
        )

        experiments[
            Description(
                name=dataset_cls
                + " 90% : Noise Augmented + Repr. Matching + Semisupervised",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.MTLDatasetsConfig(
                {"standard": dataset_large, "rep_matching": dataset_small}
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
                num_classes=180,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching Training + Semisupervised",
                representation_matching=matching_options,
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                verbose=True,
                loss_functions={"standard_img_classification": "CrossEntropyLoss"},
                **noise_type,
            ),
            seed=seed,
        )

        experiments[Description(name="transfer", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                filter_classes=(180, 200),
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                num_classes=20,
            ),
            trainer=trainer.TrainerConfig(
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                comment="Transfer + Reset on Clean",
                freeze=("core",),
                reset_linear=True,
                verbose=True,
            ),
            seed=seed,
        )

        experiments[
            Description(name="transfer noise augmented", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                filter_classes=(180, 200),
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                num_classes=20,
            ),
            trainer=trainer.TrainerConfig(
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                comment="Transfer + Reset on Noisy",
                freeze=("core",),
                reset_linear=True,
                verbose=True,
                **noise_type,
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(
                name=dataset_cls
                + " 90% : Noise Augmented + Repr. Matching + 10% Semisupervised -> Transfer 10%",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                        + " 90% : Noise Augmented + Repr. Matching + Semisupervised",
                        seed=seed,
                    )
                ],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )

        transfer_experiments[
            Description(
                name=dataset_cls
                + " 90% : Noise Augmented + Repr. Matching + 10% Semisupervised -> Transfer 10% (Noise Augmented)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls
                        + " 90% : Noise Augmented + Repr. Matching + Semisupervised",
                        seed=seed,
                    )
                ],
                experiments[Description(name="transfer noise augmented", seed=seed)],
            ]
        )

        transfer_experiments[
            Description(name=dataset_cls + " 90% : Clean -> Transfer 10%", seed=seed)
        ] = TransferExperiment(
            [
                experiments[Description(name=dataset_cls + " 90% : Clean", seed=seed)],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls + " 90% : Noise Augmented -> Transfer 10%", seed=seed
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(name=dataset_cls + " 90% : Noise Augmented", seed=seed)
                ],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                + " 90% : Noise Augmented + Repr. Matching -> Transfer 10%",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + " 90% : Noise Augmented + Repr. Matching",
                        seed=seed,
                    )
                ],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )

        transfer_experiments[
            Description(
                name=dataset_cls + " 90% : Clean -> Transfer 10% (Noise Augmented)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[Description(name=dataset_cls + " 90% : Clean", seed=seed)],
                experiments[Description(name="transfer noise augmented", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                + " 90% : Noise Augmented -> Transfer 10% (Noise Augmented)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(name=dataset_cls + " 90% : Noise Augmented", seed=seed)
                ],
                experiments[Description(name="transfer noise augmented", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls
                + " 90% : Noise Augmented + Repr. Matching -> Transfer 10% (Noise Augmented)",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls + " 90% : Noise Augmented + Repr. Matching",
                        seed=seed,
                    )
                ],
                experiments[Description(name="transfer noise augmented", seed=seed)],
            ]
        )
