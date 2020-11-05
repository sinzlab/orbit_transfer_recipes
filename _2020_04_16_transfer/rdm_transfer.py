from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}


for dataset_cls in (
    "CIFAR100",
    # "CIFAR10",
    # "TinyImageNet",
):
    for seed in (42,):
        # Noise augmentation:
        matching_options = {
            "representation": "conv_rep",
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

        experiments[Description(name=dataset_cls+": Clean", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                verbose=True,
            ),
            seed=seed,
        )

        experiments[Description(name=dataset_cls+": Noise Augmented", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented",
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                verbose=True,
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                **noise_type,
            ),
            seed=seed,
        )

        experiments[Description(name=dataset_cls+": Noise Augmented + Repr. Matching", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching",
                representation_matching=matching_options,
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                patience=1000,
                verbose=True,
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
                comment="Transfer + Reset",
                freeze=("core",),
                reset_linear=True,
                verbose=True,
            ),
            seed=seed,
        )

        experiments[Description(name="rdm_transfer", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=False,
                add_corrupted_test=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            ),
            trainer=trainer.TrainerConfig(
                max_iter=90 if dataset_cls == "TinyImageNet" else 200,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                patience=1000,
                restore_best=False,
                comment="RDM Transfer",
                verbose=True,
                rdm_transfer=True,
                rdm_prediction={"lambda": 1.0},
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(name=dataset_cls + ": Clean -> Transfer", seed=seed)
        ] = TransferExperiment(
            [
                experiments[Description(name=dataset_cls+": Clean", seed=seed)],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(name=dataset_cls + ": Noise Augmented -> Transfer", seed=seed)
        ] = TransferExperiment(
            [
                experiments[Description(name=dataset_cls+": Noise Augmented", seed=seed)],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls + ": Noise Augmented + Repr. Matching -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[Description(name=dataset_cls+": Noise Augmented + Repr. Matching", seed=seed)],
                experiments[Description(name="transfer", seed=seed)],
            ]
        )

        transfer_experiments[
            Description(name=dataset_cls + ": Clean -> RDM Transfer", seed=seed)
        ] = TransferExperiment(
            [
                experiments[Description(name=dataset_cls+": Clean", seed=seed)],
                experiments[Description(name="rdm_transfer", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls + ": Noise Augmented -> RDM Transfer", seed=seed
            )
        ] = TransferExperiment(
            [
                experiments[Description(name=dataset_cls+": Noise Augmented", seed=seed)],
                experiments[Description(name="rdm_transfer", seed=seed)],
            ]
        )
        transfer_experiments[
            Description(
                name=dataset_cls + ": Noise Augmented + Repr. Matching -> RDM Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[Description(name=dataset_cls+": Noise Augmented + Repr. Matching", seed=seed)],
                experiments[Description(name="rdm_transfer", seed=seed)],
            ]
        )
