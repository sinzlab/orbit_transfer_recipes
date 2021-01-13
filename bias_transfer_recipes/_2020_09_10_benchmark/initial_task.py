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
        scheduler_options={"milestones": (30, 60)},
        scheduler="manual",
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
for bias in (
    ("clean",),
    ("noise",),
    ("translation",),
    ("rotation_regression",),
):
    for o, t_opts in (
        (
            "default",
            {
                "max_iter": 100,
                "scheduler_options": {"milestones": (30, 60)},
                "scheduler": "manual",
            },
        ),
        (
            "long",
            {
                "max_iter": 400,
                "scheduler_options": {"milestones": (100, 200, 300)},
                "scheduler": "manual",
            },
        ),
        (
            "long_high_lr",
            {
                "max_iter": 400,
                "scheduler_options": {"milestones": (100, 200, 300)},
                "scheduler": "manual",
                "optimizer_options": {"amsgrad": False, "lr": 0.001},
            },
        ),
        (
            "long_sgd",
            {
                "max_iter": 400,
                "scheduler_options": {"milestones": (100, 200, 300)},
                "scheduler": "manual",
                "optimizer": "SGD",
                "optimizer_options": {"lr": 0.001},
            },
        ),
        (
            "long_sgd_momentum",
            {
                "max_iter": 400,
                "scheduler_options": {"milestones": (100, 200, 300)},
                "scheduler": "manual",
                "optimizer": "SGD",
                "optimizer_options": {"lr": 0.001, "momentum": 0.0001},
            },
        ),
        (
            "long_sgd_l2",
            {
                "max_iter": 400,
                "scheduler_options": {"milestones": (100, 200, 300)},
                "scheduler": "manual",
                "optimizer": "SGD",
                "optimizer_options": {"lr": 0.001, "weight_decay": 0.00001},
            },
        ),
        (
            "adaptive",
            {"max_iter": 400,
             "scheduler": "adaptive",
             "scheduler_options": {},
             "patience": 10},
        ),
        (
            "adaptive_high_lr",
            {
                "max_iter": 400,
                "scheduler": "adaptive",
                "scheduler_options": {},
                "patience": 10,
                "optimizer_options": {"amsgrad": False, "lr": 0.001},
            },
        ),
        (
            "adaptive_sgd",
            {
                "max_iter": 400,
                "optimizer": "SGD",
                "optimizer_options": {"lr": 0.001},
                "scheduler": "adaptive",
                "scheduler_options": {},
                "patience": 10,
            },
        ),
        (
            "adaptive_sgd_momentum",
            {
                "max_iter": 400,
                "scheduler": "adaptive",
                "scheduler_options": {},
                "patience": 10,
                "optimizer": "SGD",
                "optimizer_options": {"lr": 0.001, "momentum": 0.0001},
            },
        ),
        (
            "adaptive_sgd_l2",
            {
                "max_iter": 400,
                "scheduler": "adaptive",
                "scheduler_options": {},
                "patience": 10,
                "optimizer": "SGD",
                "optimizer_options": {"lr": 0.001, "weight_decay": 0.00001},
            },
        ),
    ):
        ####### step 1
        trainer_config_cls = (
            trainer.Regression
            if "regression" in bias[0]
            else trainer.TrainerConfig
        )
        experiments[
            Description(name=f"{dataset_sub_cls}-IB {bias[0]} ({o})", seed=seed,)
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
                input_channels=1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
            ),
            trainer=trainer_config_cls(
                comment=f"{dataset_sub_cls}-IB {bias[0]}",
                baseline=baseline.trainer,
                loss_functions={"regression": "CircularDistanceLoss"}
                if "regression" in bias[0]
                else {"img_classification": "CrossEntropyLoss"},
                **t_opts,
            ),
            seed=seed,
        )

        transfer_experiments[
            Description(name=f"Direct Training {bias[0]} ({o})", seed=seed,)
        ] = TransferExperiment(
            [
                experiments[
                    Description(
                        name=f"{dataset_sub_cls}-IB {bias[0]} ({o})", seed=seed,
                    )
                ]
            ]
        )
