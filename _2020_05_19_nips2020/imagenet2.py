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

        experiments[Description(name="noisy SGD 0.001", seed=seed)] = Experiment(
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
                max_iter=10,
                # lr_milestones=(30,),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=True,
                restore_best=True,
                patience=4,
                verbose=True,
                **noise_type
            ),
            seed=seed,
        )

        experiments[Description(name="noisy SGD 0.01", seed=seed)] = Experiment(
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
                max_iter=10,
                # lr_milestones=(30,),
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=True,
                restore_best=True,
                patience=4,
                verbose=True,
                **noise_type
            ),
            seed=seed,
        )

        experiments[Description(name="noisy Amsgrad 0.0003", seed=seed)] = Experiment(
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
                max_iter=10,
                optimizer="Adam",
                optimizer_options={
                    "amsgrad": True,
                    "lr": 0.0003,
                    "weight_decay": 5e-4,
                },
                lr_decay=0.8,
                adaptive_lr=True,
                restore_best=True,
                patience=4,
                verbose=True,
                **noise_type
            ),
            seed=seed,
        )


        experiments[Description(name="noisy Amsgrad 0.0003 + Warmup", seed=seed)] = Experiment(
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
                max_iter=10,
                optimizer="Adam",
                optimizer_options={
                    "amsgrad": True,
                    "lr": 0.0003,
                    "weight_decay": 5e-4,
                },
                lr_decay=0.8,
                adaptive_lr=True,
                restore_best=True,
                patience=4,
                lr_warmup=4,
                verbose=True,
                **noise_type
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(name=dataset_cls + ": Noise Augmented SGD 0.001", seed=seed)
        ] = TransferExperiment([experiments[Description(name="noisy SGD 0.001", seed=seed)],])

        transfer_experiments[
            Description(name=dataset_cls + ": Noise Augmented: SGD 0.01", seed=seed)
        ] = TransferExperiment([experiments[Description(name="noisy SGD 0.01", seed=seed)],])

        transfer_experiments[
            Description(name=dataset_cls + ": Noise Augmented: Amsgrad 0.0003", seed=seed)
        ] = TransferExperiment([experiments[Description(name="noisy Amsgrad 0.0003", seed=seed)], ])

        transfer_experiments[
            Description(name=dataset_cls + ": Noise Augmented: Amsgrad 0.0003 + Warmup", seed=seed)
        ] = TransferExperiment([experiments[Description(name="noisy Amsgrad 0.0003 + Warmup", seed=seed)], ])
