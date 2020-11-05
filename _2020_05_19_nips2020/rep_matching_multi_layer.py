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
        23,
        8
    ):

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

        match_layers = {
            # "blocks": {
            #     "conv1": "layer0.conv1",
            #     # layer1
            #     "layer1.1.conv2": "layer1.1.conv2",
            #     # layer2
            #     "layer2.1.conv2": "layer2.1.conv2",
            #     # layer3
            #     "layer3.1.conv2": "layer3.1.conv2",
            #     # layer4
            #     "layer4.1.conv2": "layer4.1.conv2",
            #     # core output
            # },
            # "logits": {
            #     # core output
            #     "fc": "readout",
            # },
            # "core": {
            #     # core output
            #     "flatten": "core",
            # },
            "all": {
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
            },
            # "all core": {
            #     "conv1": "layer0.conv1",
            #     # layer1
            #     "layer1.0.conv1": "layer1.0.conv1",
            #     "layer1.0.conv2": "layer1.0.conv2",
            #     "layer1.1.conv1": "layer1.1.conv1",
            #     "layer1.1.conv2": "layer1.1.conv2",
            #     # layer2
            #     "layer2.0.conv1": "layer2.0.conv1",
            #     "layer2.0.conv2": "layer2.0.conv2",
            #     "layer2.1.conv1": "layer2.1.conv1",
            #     "layer2.1.conv2": "layer2.1.conv2",
            #     # layer3
            #     "layer3.0.conv1": "layer3.0.conv1",
            #     "layer3.0.conv2": "layer3.0.conv2",
            #     "layer3.1.conv1": "layer3.1.conv1",
            #     "layer3.1.conv2": "layer3.1.conv2",
            #     # layer4
            #     "layer4.0.conv1": "layer4.0.conv1",
            #     "layer4.0.conv2": "layer4.0.conv2",
            #     "layer4.1.conv1": "layer4.1.conv1",
            #     "layer4.1.conv2": "layer4.1.conv2",
            #     # core output
            # },
        }
        combine_options = ("avg",
                           "lin",
                           "exp",
                           )
        for combine in combine_options:
            for select, layers in match_layers.items():
                matching_options = {
                    "representations": list(layers.values()),
                    "criterion": "mse",
                    "combine_losses": combine,
                    "second_noise_std": {(0, 0.5): 1.0},
                    "lambda": 1.0,
                    "only_for_clean": True,
                }
                experiments[
                    Description(
                        name=dataset_cls + ": Noise Augmented + Repr Matching ({},{})".format(select,combine), seed=seed,
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
                        get_intermediate_rep=layers,
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

                transfer_experiments[
                    Description(
                        name=dataset_cls
                        + ": Noise Augmented + Repr Matching ({},{}) -> Transfer".format(
                            select, combine
                        ),
                        seed=seed,
                    )
                ] = TransferExperiment(
                    [
                        experiments[
                            Description(
                                name=dataset_cls
                                + ": Noise Augmented + Repr Matching ({},{})".format(
                                    select, combine
                                ),
                                seed=seed,
                            )
                        ],
                        transfer_normal,
                    ]
                )
