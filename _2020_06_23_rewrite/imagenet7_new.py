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

        experiments[
            Description(name="Noise Augmented SGD 0.001", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                use_c_test_as_val=True,
                batch_size=200,
                num_workers=16,
                download=True,
                pin_memory=False,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls, dataset_cls=dataset_cls, pretrained=True
            ),
            trainer=trainer.TrainerConfig(
                comment="SGD 0.001",
                max_iter=90,
                # lr_milestones=(30,),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                min_lr=0.000001,
                adaptive_lr=True,
                restore_best=True,
                patience=8,
                lr_decay_steps=2,
                apply_noise_to_validation=False,
                verbose=True,
                **noise_type
            ),
            seed=seed,
        )

        experiments[
            Description(name="Noise Augmented SGD 0.01", seed=seed)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=False,
                use_c_test_as_val=False,
                batch_size=200,
                num_workers=8,
                download=True,
                pin_memory=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls, dataset_cls=dataset_cls, pretrained=True
            ),
            trainer=trainer.TrainerConfig(
                comment="SGD 0.01",
                max_iter=90,
                # lr_milestones=(30,),
                optimizer="SGD",
                optimizer_options={"lr": 0.01, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=True,
                restore_best=True,
                patience=8,
                lr_decay_steps=2,
                apply_noise_to_validation=False,
                min_lr=0.000001,
                verbose=True,
                **noise_type
            ),
            seed=seed,
        )

        # experiments[Description(name="noisy Adam 0.0001", seed=seed)] = Experiment(
        #     dataset=dataset.ImageDatasetConfig(
        #         comment=dataset_cls,
        #         dataset_cls=dataset_cls,
        #         apply_data_normalization=True,
        #         add_corrupted_test=True,
        #         batch_size=100,
        #         num_workers=8,
        #         download=False,
        #     ),
        #     model=model.ClassificationModelConfig(
        #         comment=dataset_cls, dataset_cls=dataset_cls, pretrained=True
        #     ),
        #     trainer=trainer.TrainerConfig(
        #         comment="Adam 0.0001",
        #         max_iter=90,
        #         optimizer="Adam",
        #         optimizer_options={
        #             "amsgrad": False,
        #             "lr": 0.0001,
        #             "weight_decay": 5e-4,
        #         },
        #         lr_decay=0.8,
        #         adaptive_lr=True,
        #         restore_best=True,
        #         min_lr=0.000001,
        #         patience=8,
        #         lr_decay_steps=2,
        #         verbose=True,
        #         **noise_type
        #     ),
        #     seed=seed,
        # )

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
                max_iter=10,
                lr_milestones=(5,),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                patience=10000,
                lr_decay_steps=10000,
                min_lr=0.000001,
                verbose=True,
                freeze=("core",),
                reset_linear=True,
            ),
            seed=seed,
        )

        experiments[Description(name="download", seed=seed)] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                apply_data_normalization=True,
                add_corrupted_test=True,
                use_c_test_as_val=False,
                batch_size=200,
                num_workers=16,
                pin_memory=False,
                download=True,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls,
                dataset_cls=dataset_cls,
                pretrained=True,
                pretrained_url="https://github.com/bethgelab/game-of-noise/releases/download/v1.0/Gauss_mult_Model.pth",
            ),
            trainer=trainer.TrainerConfig(
                comment="Download",
                max_iter=0,
                lr_milestones=(5,),
                optimizer="SGD",
                optimizer_options={"lr": 0.001, "momentum": 0.9},
                lr_decay=0.1,
                adaptive_lr=False,
                restore_best=False,
                patience=10000,
                lr_decay_steps=10000,
                min_lr=0.000001,
                verbose=True,
            ),
            seed=seed,
        )

        # transfer_experiments[
        #     Description(
        #         name="Noise Augmented SGD 0.001 -> Transfer", seed=seed
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[Description(name="Noise Augmented SGD 0.001", seed=seed)],
        #         experiments[Description(name="transfer SGD 0.001", seed=seed)],
        #     ]
        # )

        transfer_experiments[
            Description(name="Game of Noise", seed=seed)
        ] = TransferExperiment(
            [
                experiments[Description(name="download", seed=seed)],
            ]
        )
        # transfer_experiments[
        #     Description(name="Noise Augmented: SGD 0.01 -> Transfer", seed=seed)
        # ] = TransferExperiment(
        #     [
        #         experiments[Description(name="Noise Augmented SGD 0.01", seed=seed)],
        #         experiments[Description(name="transfer SGD 0.001", seed=seed)],
        #     ]
        # )
        #
        # transfer_experiments[
        #     Description(
        #         name=dataset_cls + ": Noise Augmented: Adam 0.0001 -> Transfer",
        #         seed=seed,
        #     )
        # ] = TransferExperiment(
        #     [
        #         experiments[Description(name="noisy Adam 0.0001", seed=seed)],
        #         experiments[Description(name="transfer SGD 0.001", seed=seed)],
        #     ]
        # )
