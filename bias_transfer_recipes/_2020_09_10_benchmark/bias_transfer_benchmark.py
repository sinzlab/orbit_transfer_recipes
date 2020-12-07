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
for dataset_sub_cls in ("FashionMNIST",):  # "MNIST",
    for bias in (
        ("clean", "color", "color_shuffle"),
        ("noise", "clean", "noise"),
        ("translation", "clean", "translation"),
        ("rotation_regression", "clean", "rotation"),
    ):
        for transfer_method in (
            "L2",
            "Mixup",
            "L2-SP",
            "Freeze",
            "Finetune",
            "Dropout",
            "RDL",
            "KnowledgeDistillation",
        ):
            if transfer_method == "KnowledgeDistillation" and "regression" in bias[0]:
                continue

            ####### step 1
            trainer_config_cls = (
                trainer.RegressionTrainerConfig
                if "regression" in bias[0]
                else trainer.TrainerConfig
            )
            experiments[
                Description(name=f"{dataset_sub_cls}-IB {bias[0]}", seed=seed,)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=f"MNIST-IB {bias[0]}",
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
                trainer=trainer_config_cls(baseline=baseline.trainer,),
                seed=seed,
            )
            ######## step 2: generating data (optional)
            trainer_options = {
                "L2": {
                    "optimizer_options": {
                        "amsgrad": False,
                        "lr": 0.0003,
                        "weight_decay": 0.01,
                    },
                },
                "Freeze": {"freeze": ("core",)},
                "Finetune": {},
                "Dropout": {},
                "Mixup": {"regularization": {"regularizer": "Mixup", "alpha": 0.5}},
                "L2-SP": {
                    "regularization": {
                        "regularizer": "L2SP",
                        "alpha": 1.0,
                        "ignore_layers": ("fc3",) if "regression" in bias[0] else (),
                    }
                },
                "RDL": {
                    "reset": "all",
                    "single_input_stream": False,
                    "regularization": {
                        "regularizer": "RDL",
                        "alpha": 1.0,
                        "dist_measure": "corr",
                        "decay_alpha": False,
                    },
                },
                "KnowledgeDistillation": {
                    "reset": "all",
                    "single_input_stream": False,
                    "regularization": {
                        "regularizer": "KnowledgeDistillation",
                        "alpha": 1.0,
                        "decay_alpha": False,
                        "softmax_temp": 10.0,
                    },
                },
            }
            if transfer_method == "KnowledgeDistillation":
                get_rep = {"fc3": "fc3"}
            elif transfer_method == "RDL":
                get_rep = {
                    "fc3": "fc3",
                }
                if bias[0] == "translation":
                    get_rep["fc2"] = "core"
                else:
                    get_rep["conv2"] = "core"
            else:
                get_rep = {}

            dataset_config_cls = dataset.ImageDatasetConfig
            transfer_config_cls = (
                trainer.TransferTrainerRegressionConfig
                if "regression" in bias[0]
                else trainer.TransferTrainerConfig
            )
            if transfer_method in ("RDL", "KnowledgeDistillation"):
                experiments[
                    Description(
                        name=f"{dataset_sub_cls} Data Generation ({transfer_method}) {bias[0]}",
                        seed=seed,
                    )
                ] = Experiment(
                    dataset=dataset.ImageDatasetConfig(
                        baseline=baseline.dataset,
                        dataset_sub_cls=dataset_sub_cls,
                        bias=bias[0],
                        shuffle=False,
                        valid_size=0.0,
                        convert_to_rgb=("color" in bias[1]),
                    ),
                    model=model.ClassificationModelConfig(
                        baseline=baseline.model,
                        bias=bias[0],
                        type="lenet300-100" if bias[0] == "translation" else "lenet5",
                        input_channels=3 if "color" in bias[1] else 1,
                        get_intermediate_rep=get_rep,
                    ),
                    trainer=transfer_config_cls(
                        baseline=baseline.trainer, save_input=True,
                    ),
                    seed=seed,
                )
                dataset_config_cls = dataset.TransferredDatasetConfig

            ######## step 3: transfer
            experiments[
                Description(
                    name=f"{dataset_sub_cls} Transfer ({transfer_method}) {bias[1]}",
                    seed=seed,
                )
            ] = Experiment(
                dataset=dataset_config_cls(
                    baseline=baseline.dataset,
                    dataset_sub_cls=dataset_sub_cls,
                    bias=bias[1],
                ),
                model=model.ClassificationModelConfig(
                    baseline=baseline.model,
                    bias=bias[1],
                    input_channels=3 if "color" in bias[1] else 1,
                    type="lenet300-100" if bias[0] == "translation" else "lenet5",
                    dropout=0.5 if transfer_method == "Dropout" else 0.0,
                    get_intermediate_rep=get_rep,
                ),
                trainer=trainer.TrainerConfig(
                    baseline=baseline.trainer,
                    readout_name="fc3",
                    **trainer_options[transfer_method],
                ),
                seed=seed,
            )
            ##### step 4: eval
            experiments[
                Description(name=f"Test {dataset_sub_cls}-IB {bias[2]}", seed=seed,)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=f"MNIST-IB {bias[2]}",
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
                trainer=trainer.TrainerConfig(baseline=baseline.trainer, max_iter=0),
                seed=seed,
            )

            transfer_experiments[
                Description(name=f"Transfer ({transfer_method}) ({bias[0]}->{bias[1]};{bias[2]})", seed=seed,)
            ] = TransferExperiment(
                [
                    experiments[
                        Description(name=f"{dataset_sub_cls}-IB {bias[0]}", seed=seed,)
                    ]
                ]
                + (
                    [
                        experiments[
                            Description(
                                name=f"{dataset_sub_cls} Data Generation ({transfer_method}) {bias[0]}",
                                seed=seed,
                            )
                        ],
                    ]
                    if transfer_method in ("RDL", "KnowledgeDistillation")
                    else []
                )
                + [
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

        ####### Direct training on bias[1]
        experiments[
            Description(name=f"{dataset_sub_cls} Direct Training {bias[1]}", seed=seed,)
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
            ),
            trainer=trainer.TrainerConfig(baseline=baseline.trainer,),
            seed=seed,
        )

        transfer_experiments[
            Description(name=f"Direct Training A ({bias[0]}->{bias[1]};{bias[2]})", seed=seed,)
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=f"{dataset_sub_cls} Direct Training {bias[1]}", seed=seed,
                    )
                ],
                experiments[
                    Description(name=f"Test {dataset_sub_cls}-IB {bias[2]}", seed=seed,)
                ],
            ]
        )

        ####### Direct training on bias[2]
        experiments[
            Description(name=f"{dataset_sub_cls} Direct Training {bias[2]}", seed=seed,)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                baseline=baseline.dataset,
                dataset_sub_cls=dataset_sub_cls,
                bias=bias[2],
            ),
            model=model.ClassificationModelConfig(
                baseline=baseline.model,
                bias=bias[2],
                input_channels=3 if "color" in bias[2] else 1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
            ),
            trainer=trainer.TrainerConfig(baseline=baseline.trainer,),
            seed=seed,
        )

        experiments[
            Description(name=f"Test {dataset_sub_cls}-IB {bias[1]}", seed=seed,)
        ] = Experiment(
            dataset=dataset.ImageDatasetConfig(
                comment=f"MNIST-IB {bias[1]}",
                dataset_cls="MNIST-IB",
                dataset_sub_cls=dataset_sub_cls,
                bias=bias[1],
                baseline=baseline.dataset,
            ),
            model=model.ClassificationModelConfig(
                dataset_cls="MNIST-IB",
                bias=bias[1],
                baseline=baseline.model,
                input_channels=3 if "color" in bias[1] else 1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
            ),
            trainer=trainer.TrainerConfig(baseline=baseline.trainer, max_iter=0),
            seed=seed,
        )

        transfer_experiments[
            Description(name=f"Direct Training B ({bias[2]};{bias[0]}->{bias[1]})", seed=seed,)
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=f"{dataset_sub_cls} Direct Training {bias[2]}", seed=seed,
                    )
                ],
                experiments[
                    Description(name=f"Test {dataset_sub_cls}-IB {bias[1]}", seed=seed,)
                ],
            ]
        )
