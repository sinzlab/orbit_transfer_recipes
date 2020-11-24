from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

for dataset_cls in (
    # "CIFAR100",
    # "CIFAR10",
    ("SVHN", "MNIST",),
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
            "representations": ["block1", "core", "readout"],
            "combine_losses": "avg",
            "criterion": "mse",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
        }
        experiments[
            Description(
                name=dataset_cls[0]
                + ": Noise Augmented + Repr Matching (Euclid; clean only; 9)",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls[0],
                dataset_cls=dataset_cls[0],
                apply_data_normalization=False,
                add_corrupted_test=False,
                download=True,
                batch_size=128,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls[0],
                dataset_cls=dataset_cls[0],
                type="resnet18",
                conv_stem_kernel_size=5 if dataset_cls[0] == "TinyImageNet" else 3,
                # representation_matching=True,
                noise_adv_regression=True,
                # get_intermediate_rep={
                #     "layer1.1.relu": "block1",
                #     "flatten": "core",
                #     "fc": "readout",
                # },
            ),
            trainer=trainer.TrainerConfig(
                comment="Noise Augmented + Repr. Matching (Euclid; clean only)!",
                max_iter=90 if dataset_cls[0] == "TinyImageNet" else 10,
                lr_milestones=(30, 60)
                if dataset_cls == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                patience=1000,
                # representation_matching=matching_options,
                noise_adv_regression=True,
                apply_noise_to_validation=False,
                **noise_type
            ),
            seed=seed,
        )
        transfer = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                apply_data_normalization=False,
                add_corrupted_test=False,
                use_c_test_as_val=False,
                download=True,
                batch_size=256,
            ),
            model=model.ClassificationModelConfig(
                comment=dataset_cls[1],
                dataset_cls=dataset_cls[1],
                type="resnet18",
                conv_stem_kernel_size=5 if dataset_cls[1] == "TinyImageNet" else 3,
                # representation_matching=True,
            ),
            trainer=trainer.TrainerConfig(
                comment="Transfer + Reset",
                max_iter=90 if dataset_cls[1] == "TinyImageNet" else 1,
                lr_milestones=(30, 60)
                if dataset_cls[1] == "TinyImageNet"
                else (60, 120, 160),
                adaptive_lr=False,
                restore_best=False,
                early_stop=False,
                # freeze=("core",),
                # freeze_bn=True,
                # reset_linear=True,
                # readout_name = "fc",
                # representation_matching=matching_options,
                eval_with_bn_train=True,
                patience=1000,
                noise_test={},
            ),
            seed=seed,
        )
        transfer_experiments[
            Description(
                name=dataset_cls[1]
                + ": Noise Augmented + Repr Matching (Euclid; clean only; 9) -> Transfer",
                seed=seed,
            )
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=dataset_cls[0]
                        + ": Noise Augmented + Repr Matching (Euclid; clean only; 9)",
                        seed=seed,
                    )
                ],
                transfer,
            ]
        )
