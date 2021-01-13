from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}
baseline = Experiment(
    dataset=dataset.ImageDatasetConfig(
        comment=f"Baseline",
        dataset_cls="MNIST-IB",
        apply_data_normalization=False,
        apply_data_augmentation=False,
        add_corrupted_test=False,
        download=True,
        batch_size=128,
        convert_to_rgb=False,
        bias="clean",
    ),
    model=model.ClassificationModelConfig(
        comment="", dataset_cls="MNIST-IB", type="lenet5", bias="clean"
    ),
    trainer=trainer.TrainerConfig(
        comment="",
        max_iter=100,
        lr_milestones=(30, 60),
        adaptive_lr=False,
        restore_best=True,
        early_stop=False,
        patience=1000,
        optimizer_options={"amsgrad": False, "lr": 0.0003},
        noise_test={},
    ),
    seed=42,
)


seed = 42
dataset_sub_cls = "FashionMNIST"
for bias in (("rotation_regression", "clean", "rotation"),):
    for transfer_method in (
        ("Freeze"),
        ("Finetune"),
        ("L2-SP"),
    ):
        for arctanh in (True, False):
            ####### step 1
            trainer_config_cls = (
                trainer.Regression
                if "regression" in bias[0]
                else trainer.TrainerConfig
            )
            experiments[
                Description(
                    name=f"{dataset_sub_cls}-IB {bias[0]} arctanh={arctanh}", seed=seed,
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=f"{dataset_sub_cls}-IB {bias[0]}",
                    dataset_cls="MNIST-IB",
                    dataset_sub_cls=dataset_sub_cls,
                    bias=bias[0],
                    baseline=baseline.dataset,
                    convert_to_rgb=(
                        "color" in bias[1]
                    ),  # if we're transferring to color later
                ),
                model=model.ClassificationModelConfig(
                    dataset_cls="MNIST-IB",
                    bias=bias[0],
                    baseline=baseline.model,
                    # num_classes=1 if "regression" in bias[0] else 10,
                    input_channels=3 if "color" in bias[1] else 1,
                    type="lenet300-100" if bias[0] == "translation" else "lenet5",
                ),
                trainer=trainer_config_cls(
                    comment=f"{dataset_sub_cls}-IB {bias[0]} {arctanh}",
                    baseline=baseline.trainer,
                    loss_functions={"regression": "CircularDistanceLoss"}
                    if "regression" in bias[0]
                    else {"img_classification": "CrossEntropyLoss"},
                    scale_loss_with_arctanh=arctanh,
                    max_iter=400 if "regression" in bias[0] else 100,
                    lr_milestones=(100, 200, 300)
                    if "regression" in bias[0]
                    else (30, 60),
                    synaptic_intelligence_computation=transfer_method
                    == "SynapticIntelligence",
                ),
                seed=seed,
            )

            ##### step 4: eval
            experiments[
                Description(name=f"Test {dataset_sub_cls}-IB {bias[2]}", seed=seed,)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=f"{dataset_sub_cls}-IB {bias[2]}",
                    dataset_cls="MNIST-IB",
                    dataset_sub_cls=dataset_sub_cls,
                    bias=bias[2],
                    baseline=baseline.dataset,
                ),
                model=model.ClassificationModelConfig(
                    dataset_cls="MNIST-IB",
                    bias=bias[2],
                    baseline=baseline.model,
                    input_channels=3 if "color" in bias[1] else 1,
                    type="lenet300-100" if bias[0] == "translation" else "lenet5",
                ),
                trainer=trainer.TrainerConfig(
                    comment=f"Test {dataset_sub_cls}-IB {bias[2]}",
                    baseline=baseline.trainer,
                    max_iter=0,
                ),
                seed=seed,
            )

            ######## step 3: transfer
            trainer_options = {
                "Freeze": {"freeze": ("core",)},
                "Finetune": {},
                "L2-SP": {
                    "regularization": {
                        "regularizer": "ParamDistance",
                        "ignore_layers": ("fc3",) if "regression" in bias[0] else (),
                    }
                },
            }
            experiments[
                Description(
                    name=f"{dataset_sub_cls} Transfer ({transfer_method}) {bias[1]}",
                    seed=seed,
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    baseline=baseline.dataset,
                    dataset_sub_cls=dataset_sub_cls,
                    bias=bias[1],
                ),
                model=model.ClassificationModelConfig(
                    baseline=baseline.model,
                    bias=bias[1],
                    input_channels=3 if "color" in bias[1] else 1,
                    type="lenet300-100" if bias[0] == "translation" else "lenet5",
                    add_buffer=("importance",)
                    if transfer_method in ("EWC", "SynapticIntelligence")
                    else (),
                ),
                trainer=trainer.TrainerConfig(
                    comment=f"{dataset_sub_cls} Transfer ({transfer_method}) {bias[1]}",
                    baseline=baseline.trainer,
                    readout_name="fc3",
                    **trainer_options[transfer_method],
                ),
                seed=seed,
            )

            transfer_experiments[
                Description(
                    name=f"{transfer_method} ({bias[0]}->{bias[1]};{bias[2]}) arctanh={arctanh}",
                    seed=seed,
                )
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name=f"{dataset_sub_cls}-IB {bias[0]} arctanh={arctanh}",
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name=f"{dataset_sub_cls} Transfer ({transfer_method}) {bias[1]}",
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name=f"Test {dataset_sub_cls}-IB {bias[2]}", seed=seed,
                        )
                    ],
                ]
            )

    ####### Rotated classification -> Regression
    experiments[
        Description(name=f"{dataset_sub_cls} Training {bias[2]}", seed=seed,)
    ] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            baseline=baseline.dataset, dataset_sub_cls=dataset_sub_cls, bias=bias[2],
        ),
        model=model.ClassificationModelConfig(
            baseline=baseline.model,
            bias=bias[2],
            input_channels=3 if "color" in bias[2] else 1,
            type="lenet300-100" if bias[0] == "translation" else "lenet5",
        ),
        trainer=trainer.TrainerConfig(
            baseline=baseline.trainer,
            comment=f"{dataset_sub_cls} Direct Training {bias[2]}",
        ),
        seed=seed,
    )


    trainer_config_cls = (
        trainer.Regression
        if "regression" in bias[0]
        else trainer.TrainerConfig
    )
    experiments[
        Description(name=f"Transfer {dataset_sub_cls}-IB {bias[2]}->{bias[0]}", seed=seed,)
    ] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=f"{dataset_sub_cls}-IB {bias[0]}",
            dataset_cls="MNIST-IB",
            dataset_sub_cls=dataset_sub_cls,
            bias=bias[0],
            baseline=baseline.dataset,
            convert_to_rgb=("color" in bias[1]),  # if we're transferring to color later
        ),
        model=model.ClassificationModelConfig(
            dataset_cls="MNIST-IB",
            bias=bias[0],
            baseline=baseline.model,
            # num_classes=1 if "regression" in bias[0] else 10,
            input_channels=3 if "color" in bias[1] else 1,
            type="lenet300-100" if bias[0] == "translation" else "lenet5",
        ),
        trainer=trainer_config_cls(
            comment=f"{dataset_sub_cls}-IB {bias[0]} {arctanh}",
            baseline=baseline.trainer,
            loss_functions={"regression": "CircularDistanceLoss"}
            if "regression" in bias[0]
            else {"img_classification": "CrossEntropyLoss"},
            scale_loss_with_arctanh=False,
            max_iter=400 if "regression" in bias[0] else 100,
            lr_milestones=(100, 200, 300) if "regression" in bias[0] else (30, 60),
            synaptic_intelligence_computation=transfer_method == "SynapticIntelligence",
            freeze=("core",),
            readout_name="fc3",
        ),
        seed=seed,
    )

    transfer_experiments[
        Description(
            name=f"Transfer ({bias[2]}->{bias[0]})", seed=seed,
        )
    ] = TransferExperiment(
        [
            experiments[
                Description(
                    name=f"{dataset_sub_cls} Training {bias[2]}", seed=seed,
                )
            ],
            experiments[
                Description(name=f"Transfer {dataset_sub_cls}-IB {bias[2]}->{bias[0]}", seed=seed,)
            ],
        ]
    )
