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
        max_iter=1,
        lr_milestones=(30, 60),
        adaptive_lr=False,
        restore_best=False,
        early_stop=False,
        patience=1000,
        noise_test={},
    ),
    seed=42,
)
for bias in (
    # "color",
    "noise",
    # "translation",
    # "addition",
):
    seed = 42
    experiments[Description(name=f"MNIST-IB {bias}", seed=seed,)] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=f"MNIST-IB {bias}",
            dataset_cls="MNIST-IB",
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
    transfer = Experiment(
        dataset=dataset.ImageDatasetConfig(
            baseline=baseline.dataset,
            convert_to_rgb=(bias == "color"),
            shuffle=False,
            valid_size=0.0,
        ),
        model=model.ClassificationModelConfig(
            baseline=baseline.model, input_channels=3 if bias == "color" else 1,
        ),
        trainer=trainer.TransferConfig(
            baseline=baseline.trainer, rdm_generation="core"
        ),
        seed=seed,
    )
    experiments[Description(name=f"MNIST", seed=seed,)] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            baseline=baseline.dataset, convert_to_rgb=(bias == "color"),
        ),
        model=model.ClassificationModelConfig(
            baseline=baseline.model,
            input_channels=3 if bias == "color" else 1,
            # dropout=0.4,
        ),
        trainer=trainer.TrainerConfig(
            baseline=baseline.trainer,
            # freeze=("core",),
            reset_linear=False,
            readout_name="fc3",
            # l2sp=1.0,
            mixup=1.0,
        ),
        seed=seed,
    )
    experiments[Description(name=f"Test MNIST-IB {bias}", seed=seed,)] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=f"MNIST-IB {bias}",
            dataset_cls="MNIST-IB",
            bias=bias,
            baseline=baseline.dataset,
            convert_to_rgb=False,
        ),
        model=model.ClassificationModelConfig(
            dataset_cls="MNIST-IB",
            bias=bias,
            baseline=baseline.model,
            input_channels=3 if bias == "color" else 1,
        ),
        trainer=trainer.TrainerConfig(baseline=baseline.trainer, max_iter=0),
        seed=seed,
    )
    transfer_experiments[Description(name="Transfer", seed=seed,)] = TransferExperiment(
        [
            experiments[Description(name=f"MNIST-IB {bias}", seed=seed,)],
            transfer,
            experiments[Description(name=f"MNIST", seed=seed,)],
            # experiments[Description(name=f"Test MNIST-IB {bias}", seed=seed,)],
        ]
    )
