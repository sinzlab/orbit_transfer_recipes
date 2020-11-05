from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}


for dataset_cls in (
    # "CIFAR100",
    # "CIFAR10",
    # "TinyImageNet",
    "ImageNet",
):
    for seed in (42,):
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

        experiments[Description(name="clean", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=100,
                download=False,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls, dataset_cls=dataset_cls, pretrained=True
            ),
            trainer=trainer.TrainerConfig(
                comment="",
                max_iter=0,
                lr_milestones=(30,),
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

        experiments[Description(name="noisy", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=100,
                num_workers=8,
                download=False,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls, dataset_cls=dataset_cls, pretrained=True
            ),
            trainer=trainer.TrainerConfig(
                comment="",
                max_iter=60,
                lr_milestones=(30,),
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

        experiments[Description(name="noisy + rep matching", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=100,
                download=False,
                num_workers=8,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                pretrained=True,
                representation_matching=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="",
                max_iter=60,
                lr_milestones=(30,),
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

        experiments[Description(name="noisy + adv regression", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=100,
                download=False,
                num_workers=8,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                pretrained=True,
                noise_adv_regression=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="",
                max_iter=60,
                lr_milestones=(30,),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                verbose=True,
                noise_adv_regression=True,
                **noise_type
            ),
            seed=seed,
        )

        experiments[Description(name="transfer", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=100,
                download=False,
                num_workers=8,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls, dataset_cls=dataset_cls,
            ),
            trainer=trainer.TrainerConfig(
                comment="",
                max_iter=60,
                lr_milestones=(30,),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                verbose=True,
                freeze=("core",),
                reset_linear=True,
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(name=dataset_cls + ": Clean", seed=seed)
        ] = TransferExperiment(
            [
                experiments[Description(name="clean", seed=seed)],
                # experiments[Description(name="transfer", seed=seed)
                # ]
            ]
        )
        transfer_experiments[
            Description(name=dataset_cls + ": Noise Augmented", seed=seed)
        ] = TransferExperiment([experiments[Description(name="noisy", seed=seed)],
                                experiments[Description(name="transfer", seed=seed)]])
        transfer_experiments[
            Description(name=dataset_cls + ": Noise Augmented + Representation Matching", seed=seed)
        ] = TransferExperiment([experiments[Description(name="noisy + rep matching", seed=seed)],
                                experiments[Description(name="transfer", seed=seed)]])
        transfer_experiments[
            Description(name=dataset_cls + ": Noise Augmented + Adv Regression", seed=seed)
        ] = TransferExperiment([experiments[Description(name="noisy + adv regression", seed=seed)],
                                experiments[Description(name="transfer", seed=seed)]])
