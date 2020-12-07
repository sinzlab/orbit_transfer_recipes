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
        max_iter=1,
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
for bias in (("color_easy", "clean_shuffle", "color_easy"),):
    for alpha in (1.0,):
        transfer_method = "SynapticIntelligence"
        softmax_temp = 1.0
        trainer_options = {
            "L2": {
                "optimizer_options": {
                    "amsgrad": False,
                    "lr": 0.0003,
                    "weight_decay": alpha,
                },
            },
            "Freeze": {"freeze": ("core",)},
            "Finetune": {},
            "Dropout": {},
            "Mixup": {"regularization": {"regularizer": "Mixup", "alpha": alpha}},
            "L2-SP": {
                "regularization": {
                    "regularizer": "ParamDistance",
                    "alpha": alpha,
                    "ignore_layers": ("fc3",) if "regression" in bias[0] else (),
                }
            },
            "RDL": {
                "reset": "all",
                "single_input_stream": False,
                "regularization": {
                    "regularizer": "RDL",
                    "alpha": alpha,
                    "dist_measure": "corr",
                    "decay_alpha": False,
                },
            },
            "KnowledgeDistillation": {
                "reset": "all",
                "single_input_stream": False,
                "regularization": {
                    "regularizer": "KnowledgeDistillation",
                    "alpha": alpha,
                    "decay_alpha": False,
                    "softmax_temp": softmax_temp,
                },
            },
            "EWC": {
                "reset": "all",
                "regularization": {"regularizer": "ParamDistance", "alpha": alpha,},
            },
            "SynapticIntelligence": {
                "reset": "all",
                "regularization": {"regularizer": "ParamDistance", "alpha": alpha,},
            },
        }
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
                comment=f"{dataset_sub_cls}-IB {bias[0]}",
                dataset_cls="MNIST-IB",
                dataset_sub_cls=dataset_sub_cls,
                bias=bias[0],
                baseline=baseline.dataset,
            ),
            model=model.ClassificationModelConfig(
                dataset_cls="MNIST-IB",
                bias=bias[0],
                baseline=baseline.model,
                # num_classes=1 if "regression" in bias[0] else 10,
                input_channels=3 if "color" in bias[0] else 1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
            ),
            trainer=trainer_config_cls(
                comment=f"{dataset_sub_cls}-IB {bias[0]}",
                baseline=baseline.trainer,
                loss_functions={"regression": "CircularDistanceLoss"}
                if "regression" in bias[0]
                else {"img_classification": "CrossEntropyLoss"},
                synaptic_intelligence_computation=transfer_method
                == "SynapticIntelligence",
            ),
            seed=seed,
        )
        ######## step 2: generating data (optional)
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
        if transfer_method in (
            "RDL",
            "KnowledgeDistillation",
            "EWC",
            "SynapticIntelligence",
        ):
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
                ),
                model=model.ClassificationModelConfig(
                    baseline=baseline.model,
                    bias=bias[0],
                    type="lenet300-100" if bias[0] == "translation" else "lenet5",
                    input_channels=3 if "color" in bias[0] else 1,
                    get_intermediate_rep=get_rep,
                    add_buffer=("SI_omega", "SI_prev_task")
                    if transfer_method == "SynapticIntelligence"
                    else (),
                ),
                trainer=transfer_config_cls(
                    comment=f"{dataset_sub_cls} Data Generation ({transfer_method}) {bias[0]}",
                    baseline=baseline.trainer,
                    save_input=transfer_method in ("RDL", "KnowledgeDistillation"),
                    save_representation=transfer_method
                    in ("RDL", "KnowledgeDistillation"),
                    compute_fisher={"num_samples": 1024, "empirical": True}
                    if transfer_method == "EWC"
                    else {},
                    compute_si_omega={"damping_factor": 0.0001}
                    if transfer_method == "SynapticIntelligence"
                    else {},
                    loss_functions={"regression": "CircularDistanceLoss"}
                    if "regression" in bias[0]
                    else {"img_classification": "CrossEntropyLoss"},
                ),
                seed=seed,
            )

            print(
                experiments[
                    Description(
                        name=f"{dataset_sub_cls} Data Generation ({transfer_method}) {bias[0]}",
                        seed=seed,
                    )
                ]
            )
            if transfer_method in ("RDL", "KnowledgeDistillation"):
                dataset_config_cls = dataset.TransferredDatasetConfig

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
                input_channels=3 if "color" in bias[0] else 1,
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
        experiments[
            Description(
                name=f"{dataset_sub_cls} Transfer ({transfer_method}: {alpha}) {bias[1]}",
                seed=seed,
            )
        ] = Experiment(
            dataset=dataset_config_cls(
                baseline=baseline.dataset,
                dataset_sub_cls=dataset_sub_cls,
                bias=bias[1],
                convert_to_rgb=("color" in bias[0]),
            ),
            model=model.ClassificationModelConfig(
                baseline=baseline.model,
                bias=bias[1],
                input_channels=3 if "color" in bias[0] else 1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
                dropout=alpha if transfer_method == "Dropout" else 0.0,
                get_intermediate_rep=get_rep,
                add_buffer=("importance",)
                if transfer_method in ("EWC", "SynapticIntelligence")
                else (),
            ),
            trainer=trainer.TrainerConfig(
                comment=f"{dataset_sub_cls} Transfer ({transfer_method}: {alpha}) {bias[1]}",
                baseline=baseline.trainer,
                readout_name="fc3",
                **trainer_options[transfer_method],
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(
                name=f"Transfer {transfer_method}: {alpha} ({bias[0]}->{bias[1]};{bias[2]})",
                seed=seed,
            )
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
                if transfer_method
                in ("RDL", "KnowledgeDistillation", "EWC", "SynapticIntelligence")
                else []
            )
            + [
                experiments[
                    Description(
                        name=f"{dataset_sub_cls} Transfer ({transfer_method}: {alpha}) {bias[1]}",
                        seed=seed,
                    )
                ],
                experiments[
                    Description(name=f"Test {dataset_sub_cls}-IB {bias[2]}", seed=seed,)
                ],
            ]
        )
