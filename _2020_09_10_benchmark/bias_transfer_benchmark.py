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
        bias="expansion",
    ),
    model=model.ClassificationModelConfig(
        comment="", dataset_cls="MNIST-IB", type="lenet5", bias="expansion"
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
for dataset_sub_cls in (#"MNIST",
                         "FashionMNIST",
                        ):
    for bias in (
        "rotation",
        "color",
        "noise",
        "translation",
        # "addition",
    ):
        for transfer_method in (
            "L2",
            "Mixup",
            "L2-SP",
            "Freeze",
            "Finetune",
            "Dropout",
        ):
            experiments[
                Description(name=f"{dataset_sub_cls}-IB {bias}", seed=seed,)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=f"MNIST-IB {bias}",
                    dataset_cls="MNIST-IB",
                    dataset_sub_cls=dataset_sub_cls,
                    bias=bias,
                    baseline=baseline.dataset,
                    convert_to_rgb=False,
                ),
                model=model.ClassificationModelConfig(
                    dataset_cls="MNIST-IB",
                    bias=bias,
                    baseline=baseline.model,
                    num_classes=1 if bias == "addition" else 10,
                    input_channels=3 if bias == "color" else 1,
                    type="lenet300-100" if bias == "translation" else "lenet5",
                ),
                trainer=trainer.TrainerConfig(
                    baseline=baseline.trainer,
                    loss_functions={
                        "img_classification": "MSELoss"
                        if bias == "addition"
                        else "CrossEntropyLoss"
                    },
                ),
                seed=seed,
            )
            experiments[
                Description(name=f"Test {dataset_sub_cls}-IB {bias}", seed=seed,)
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    comment=f"MNIST-IB {bias}",
                    dataset_cls="MNIST-IB",
                    dataset_sub_cls=dataset_sub_cls,
                    bias=bias,
                    baseline=baseline.dataset,
                    convert_to_rgb=False,
                ),
                model=model.ClassificationModelConfig(
                    dataset_cls="MNIST-IB",
                    bias=bias,
                    baseline=baseline.model,
                    input_channels=3 if bias == "color" else 1,
                    type="lenet300-100" if bias == "translation" else "lenet5",
                ),
                trainer=trainer.TrainerConfig(baseline=baseline.trainer, max_iter=0),
                seed=seed,
            )

            trainer_options = {
                "L2": {
                    "optimizer_options": {
                        "amsgrad": False,
                        "lr": 0.0003,
                        "weight_decay": 0.01,
                    },
                },
                "Mixup": {"mixup": 1.0},
                "L2-SP": {"l2sp": 1.0},
                "Freeze": {"freeze": ("core",)},
                "Finetune": {},
                "Dropout": {},
            }
            experiments[
                Description(
                    name=f"{dataset_sub_cls} Transfer ({transfer_method}) {bias}",
                    seed=seed,
                )
            ] = Experiment(
                dataset=dataset.ImageDatasetConfig(
                    baseline=baseline.dataset,
                    convert_to_rgb=(bias == "color"),
                    dataset_sub_cls=dataset_sub_cls,
                ),
                model=model.ClassificationModelConfig(
                    baseline=baseline.model,
                    input_channels=3 if bias == "color" else 1,
                    type="lenet300-100" if bias == "translation" else "lenet5",
                    dropout=0.5 if transfer_method == "Dropout" else 0.0,
                ),
                trainer=trainer.TrainerConfig(
                    baseline=baseline.trainer,
                    readout_name="fc3",
                    reset_linear=False,
                    **trainer_options[transfer_method],
                ),
                seed=seed,
            )
            # transfer_experiments[
            #     Description(name=f"Transfer ({transfer_method}) {bias}", seed=seed,)
            # ] = TransferExperiment(
            #     [
            #         experiments[
            #             Description(name=f"{dataset_sub_cls}-IB {bias}", seed=seed,)
            #         ],
            #         experiments[
            #             Description(
            #                 name=f"{dataset_sub_cls} Transfer ({transfer_method}) {bias}",
            #                 seed=seed,
            #             )
            #         ],
            #         experiments[
            #             Description(
            #                 name=f"Test {dataset_sub_cls}-IB {bias}", seed=seed,
            #             )
            #         ],
            #     ]
            # )
            transfer_experiments[
                Description(name=f"No Transfer ({transfer_method}) {bias}", seed=seed,)
            ] = TransferExperiment(
                [
                    experiments[
                        Description(
                            name=f"{dataset_sub_cls} Transfer ({transfer_method}) {bias}",
                            seed=seed,
                        )
                    ],
                    experiments[
                        Description(
                            name=f"Test {dataset_sub_cls}-IB {bias}", seed=seed,
                        )
                    ],
                ]
            )
