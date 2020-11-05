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
    seed = 42
    # Noise augmentation:
    noise_type = {
        "add_noise": True,
        "noise_snr": None,
        "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5},
    }
    intermediate_outputs = {
        "conv1": "layer0.conv1",
        "relu": "layer0.relu",
        # layer1
        "layer1.0.conv1": "layer1.0.conv1",
        "layer1.0.conv2": "layer1.0.conv2",
        "layer1.0.conv3": "layer1.0.conv3",
        "layer1.0.relu": "layer1.0.relu",
        "layer1.1.conv1": "layer1.1.conv1",
        "layer1.1.conv2": "layer1.1.conv2",
        "layer1.1.conv3": "layer1.1.conv3",
        "layer1.1.relu": "layer1.1.relu",
        "layer1.2.conv1": "layer1.2.conv1",
        "layer1.2.conv2": "layer1.2.conv2",
        "layer1.2.conv3": "layer1.2.conv3",
        "layer1.2.relu": "layer1.2.relu",
        # layer2
        "layer2.0.conv1": "layer2.0.conv1",
        "layer2.0.conv2": "layer2.0.conv2",
        "layer2.0.conv3": "layer2.0.conv3",
        "layer2.0.relu": "layer2.0.relu",
        "layer2.1.conv1": "layer2.1.conv1",
        "layer2.1.conv2": "layer2.1.conv2",
        "layer2.1.conv3": "layer2.1.conv3",
        "layer2.1.relu": "layer2.1.relu",
        "layer2.2.conv1": "layer2.2.conv1",
        "layer2.2.conv2": "layer2.2.conv2",
        "layer2.2.conv3": "layer2.2.conv3",
        "layer2.2.relu": "layer2.2.relu",
        "layer2.3.conv1": "layer2.3.conv1",
        "layer2.3.conv2": "layer2.3.conv2",
        "layer2.3.conv3": "layer2.3.conv3",
        "layer2.3.relu": "layer2.3.relu",
        # layer3
        "layer3.0.conv1": "layer3.0.conv1",
        "layer3.0.conv2": "layer3.0.conv2",
        "layer3.0.conv3": "layer3.0.conv3",
        "layer3.0.relu": "layer3.0.relu",
        "layer3.1.conv1": "layer3.1.conv1",
        "layer3.1.conv2": "layer3.1.conv2",
        "layer3.1.conv3": "layer3.1.conv3",
        "layer3.1.relu": "layer3.1.relu",
        "layer3.2.conv1": "layer3.2.conv1",
        "layer3.2.conv2": "layer3.2.conv2",
        "layer3.2.conv3": "layer3.2.conv3",
        "layer3.2.relu": "layer3.2.relu",
        "layer3.3.conv1": "layer3.3.conv1",
        "layer3.3.conv2": "layer3.3.conv2",
        "layer3.3.conv3": "layer3.3.conv3",
        "layer3.3.relu": "layer3.3.relu",
        "layer3.4.conv1": "layer3.4.conv1",
        "layer3.4.conv2": "layer3.4.conv2",
        "layer3.4.conv3": "layer3.4.conv3",
        "layer3.4.relu": "layer3.4.relu",
        "layer3.5.conv1": "layer3.5.conv1",
        "layer3.5.conv2": "layer3.5.conv2",
        "layer3.5.conv3": "layer3.5.conv3",
        "layer3.5.relu": "layer3.5.relu",
        # layer4
        "layer4.0.conv1": "layer4.0.conv1",
        "layer4.0.conv2": "layer4.0.conv2",
        "layer4.0.conv3": "layer4.0.conv3",
        "layer4.0.relu": "layer4.0.relu",
        "layer4.1.conv1": "layer4.1.conv1",
        "layer4.1.conv2": "layer4.1.conv2",
        "layer4.1.conv3": "layer4.1.conv3",
        "layer4.1.relu": "layer4.1.relu",
        "layer4.2.conv1": "layer4.2.conv1",
        "layer4.2.conv2": "layer4.2.conv2",
        "layer4.2.conv3": "layer4.2.conv3",
        "layer4.2.relu": "layer4.2.relu",
        # core output
        "flatten": "core",
        "fc": "readout",
    }
    noise_adv = {"noise_adv_classification": False, "noise_adv_regression": True}

    experiments[Description(name=dataset_cls + ": Clean", seed=42)] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            apply_data_normalization=True,
            add_corrupted_test=True,
            download=True,
            batch_size=64,
        ),
        model=model.ClassificationModelConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            get_intermediate_rep=intermediate_outputs,
        ),
        trainer=trainer.TrainerConfig(
            comment="",
            max_iter=90 if dataset_cls == "TinyImageNet" else 200,
            lr_milestones=(30, 60) if dataset_cls == "TinyImageNet" else (60, 120, 160),
            adaptive_lr=False,
            restore_best=False,
            early_stop=False,
            patience=1000,
        ),
        seed=seed,
    )
    experiments[
        Description(name=dataset_cls + ": Noise Augmented", seed=42)
    ] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            apply_data_normalization=True,
            add_corrupted_test=True,
            download=True,
            batch_size=64,
        ),
        model=model.ClassificationModelConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            get_intermediate_rep=intermediate_outputs,
        ),
        trainer=trainer.TrainerConfig(
            comment="Noise Augmented",
            max_iter=90 if dataset_cls == "TinyImageNet" else 200,
            lr_milestones=(30, 60) if dataset_cls == "TinyImageNet" else (60, 120, 160),
            adaptive_lr=False,
            restore_best=False,
            early_stop=False,
            patience=1000,
            **noise_type
        ),
        seed=seed,
    )
    matching_options = {
        "representation": "core",
        "criterion": "cosine",
        "second_noise_std": {(0, 1.0): 1.0},
        "lambda": 1.0,
    }
    experiments[
        Description(name=dataset_cls + ": Noise Augmented + Repr Matching", seed=42)
    ] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            apply_data_normalization=True,
            add_corrupted_test=True,
            download=True,
            batch_size=64,
        ),
        model=model.ClassificationModelConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            get_intermediate_rep=intermediate_outputs,
            representation_matching=True,
        ),
        trainer=trainer.TrainerConfig(
            comment="Noise Augmented + Repr. Matching (Cosine)",
            max_iter=90 if dataset_cls == "TinyImageNet" else 200,
            lr_milestones=(30, 60) if dataset_cls == "TinyImageNet" else (60, 120, 160),
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
        "representation": "core",
        "criterion": "mse",
        "second_noise_std": {(0, 0.5): 1.0},
        "lambda": 1.0,
    }
    experiments[
        Description(
            name=dataset_cls + ": Noise Augmented + Repr Matching (Euclid)", seed=42
        )
    ] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            apply_data_normalization=True,
            add_corrupted_test=True,
            download=True,
            batch_size=64,
        ),
        model=model.ClassificationModelConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            get_intermediate_rep=intermediate_outputs,
            representation_matching=True,
        ),
        trainer=trainer.TrainerConfig(
            comment="Noise Augmented + Repr. Matching (Euclid)",
            max_iter=90 if dataset_cls == "TinyImageNet" else 200,
            lr_milestones=(30, 60) if dataset_cls == "TinyImageNet" else (60, 120, 160),
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
            name=dataset_cls + ": Noise Augmented + Noise Adv Regression", seed=42
        )
    ] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            apply_data_normalization=True,
            add_corrupted_test=True,
            download=True,
            batch_size=64,
        ),
        model=model.ClassificationModelConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            get_intermediate_rep=intermediate_outputs,
            noise_adv_regression=True,
        ),
        trainer=trainer.TrainerConfig(
            comment="Noise Augmented + Noise Adv Regression",
            max_iter=90 if dataset_cls == "TinyImageNet" else 200,
            lr_milestones=(30, 60) if dataset_cls == "TinyImageNet" else (60, 120, 160),
            adaptive_lr=False,
            restore_best=False,
            early_stop=False,
            patience=1000,
            noise_adv_loss_factor=1.0,
            noise_adv_gamma=10.0,
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
            batch_size=64,
        ),
        model=model.ClassificationModelConfig(
            comment=dataset_cls,
            dataset_cls=dataset_cls,
            conv_stem_kernel_size=5 if dataset_cls == "TinyImageNet" else 3,
            get_intermediate_rep=intermediate_outputs,
        ),
        trainer=trainer.TrainerConfig(
            comment="Transfer + Reset",
            max_iter=90 if dataset_cls == "TinyImageNet" else 200,
            lr_milestones=(30, 60) if dataset_cls == "TinyImageNet" else (60, 120, 160),
            adaptive_lr=False,
            restore_best=False,
            early_stop=False,
            freeze=("core",),
            reset_linear=True,
            patience=1000,
        ),
        seed=seed,
    )
    transfer_experiments[
        Description(name=dataset_cls + ": Clean", seed=42)
    ] = TransferExperiment(
        [experiments[Description(name=dataset_cls + ": Clean", seed=42)],]
    )
    transfer_experiments[
        Description(name=dataset_cls + ": Noise Augmented -> Transfer (Reset)", seed=42)
    ] = TransferExperiment(
        [
            experiments[Description(name=dataset_cls + ": Noise Augmented", seed=42)],
            transfer,
        ]
    )
    transfer_experiments[
        Description(
            name=dataset_cls
            + ": Noise Augmented + Noise Adv Regression -> Transfer (Reset)",
            seed=42,
        )
    ] = TransferExperiment(
        [
            experiments[
                Description(
                    name=dataset_cls + ": Noise Augmented + Noise Adv Regression",
                    seed=42,
                )
            ],
            transfer,
        ]
    )
    transfer_experiments[
        Description(
            name=dataset_cls
            + ": Noise Augmented + Repr Matching (Euclid) -> Transfer (Reset)",
            seed=42,
        )
    ] = TransferExperiment(
        [
            experiments[
                Description(
                    name=dataset_cls + ": Noise Augmented + Repr Matching (Euclid)",
                    seed=42,
                )
            ],
            transfer,
        ]
    )
    transfer_experiments[
        Description(
            name=dataset_cls + ": Noise Augmented + Repr Matching -> Transfer (Reset)",
            seed=42,
        )
    ] = TransferExperiment(
        [
            experiments[
                Description(
                    name=dataset_cls + ": Noise Augmented + Repr Matching", seed=42
                )
            ],
            transfer,
        ]
    )
