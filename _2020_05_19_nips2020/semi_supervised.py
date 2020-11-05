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
        supervised_data = dataset.ImageDatasetConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            apply_data_normalization=True,
            add_corrupted_test=True,
            download=True,
            batch_size=128
        )
        semisupervised_data = dataset.ImageDatasetConfig(
            comment="CIFAR10-Semisupervised",
            dataset_cls="CIFAR10-Semisupervised",
            apply_data_normalization=True,
            add_corrupted_test=False,
            download=True,
            batch_size=128
        )

        experiments[Description(name=dataset_cls + ": Clean", seed=seed)] = Experiment(
            dataset=supervised_data,
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=False,
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
            Description(name=dataset_cls + ": Noise Augmented", seed=seed)
        ] = Experiment(
            dataset=supervised_data,
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=False,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented",
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
                name=dataset_cls + ": Noise Augmented + Repr. Matching", seed=seed
            )
        ] = Experiment(
            dataset=supervised_data,
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
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
                + ": Noise Augmented + Repr. Matching + Semisupervised",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.MTLDatasetsConfig(
                {"standard": supervised_data, "rep_matching": semisupervised_data}
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
                representation_matching=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching Training + Semisupervised",
                representation_matching=matching_options,
                max_iter=40,
                lr_milestones=(20, 30)
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

        transfer_experiments = experiments
