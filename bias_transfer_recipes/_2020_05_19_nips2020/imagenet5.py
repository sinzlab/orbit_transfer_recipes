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


        experiments[
            Description(name="Noise Augmented SGD 0.01 + Adv Regression", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=100,
                num_workers=8,
                pin_memory=False,
                download=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                pretrained=True,
                noise_adv_regression=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="SGD 0.01",
                max_iter=90,
                # lr_milestones=(30,),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=True,
                restore_best=True,
                patience=8,
                lr_decay_steps=2,
                min_lr=0.000001,
                verbose=True,
                noise_adv_regression=True,
                **noise_type
            ),
            seed=seed,
        )

        experiments[Description(name="transfer SGD 0.001", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                batch_size=200,
                num_workers=16,
                pin_memory=False,
                download=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls, dataset_cls=dataset_cls, pretrained=False,
            ),
            trainer=trainer.TrainerConfig(
                comment="Transfer SGD 0.001",
                max_iter=30,
                # lr_milestones=(30,),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=True,
                restore_best=True,
                patience=8,
                lr_decay_steps=2,
                min_lr=0.000001,
                verbose=True,
                freeze=("core",),
                reset_linear=True
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(
                name="Noise Augmented: SGD 0.01 + Adv Regression -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(name="Noise Augmented SGD 0.01 + Adv Regression", seed=seed)
                ],
                experiments[Description(name="transfer SGD 0.001", seed=seed)],
            ]
        )
