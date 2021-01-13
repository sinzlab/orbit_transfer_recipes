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
    transfer_method = "VCL"
    trainer_options = {
        "VCL": {"reset": "", "regularization": {"regularizer": "VCL",},},
    }

    # 1: Train on Task A
    trainer_config_cls = (
        trainer.Regression
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
            type="lenet300-100",
        ),
        trainer=trainer_config_cls(
            comment=f"{dataset_sub_cls}-IB {bias[0]}",
            baseline=baseline.trainer,
            loss_functions={"regression": "CircularDistanceLoss"}
            if "regression" in bias[0]
            else {"img_classification": "CrossEntropyLoss"},
            synaptic_intelligence_computation=transfer_method == "SynapticIntelligence",
        ),
        seed=seed,
    )

    # 2: Extract Coreset on Task A and Load Parameters
    transfer_config_cls = (
        trainer.TransferTrainerRegressionConfig
        if "regression" in bias[0]
        else trainer.TransferTrainerConfig
    )
    experiments[
        Description(
            name=f"{dataset_sub_cls}-IB Extract Coreset ({transfer_method}) {bias[0]}",
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
            type="lenet300-100-bayes",
            input_channels=3 if "color" in bias[0] else 1,
            get_intermediate_rep={},
            add_buffer=(),
        ),
        trainer=transfer_config_cls(
            comment=f"{dataset_sub_cls}-IB Extract Coreset ({transfer_method}) {bias[0]}",
            baseline=baseline.trainer,
            extract_coreset={"method": "random", "size": 200},
            loss_functions={"regression": "CircularDistanceLoss"}
            if "regression" in bias[0]
            else {"img_classification": "CrossEntropyLoss"},
            data_transfer=True,
        ),
        seed=seed,
    )

    # 3: Train on Task A with VCL Loss
    experiments[
        Description(name=f"{dataset_sub_cls}-IB Train VCL {bias[0]}", seed=seed,)
    ] = Experiment(
        dataset=dataset.Generated(
            baseline=baseline.dataset,
            dataset_sub_cls=dataset_sub_cls,
            bias=bias[0],
            train_on_reduced_data=True,
        ),
        model=model.ClassificationModelConfig(
            baseline=baseline.model,
            bias=bias[0],
            input_channels=3 if "color" in bias[0] else 1,
            type="lenet300-100-bayes",
        ),
        trainer=trainer.TrainerConfig(
            comment=f"{dataset_sub_cls}-IB Transfer ({transfer_method}) {bias[0]}",
            baseline=baseline.trainer,
            readout_name="fc3",
            data_transfer=True,
            **trainer_options[transfer_method],
        ),
        seed=seed,
    )

    # 4: Extract and update coreset on Task B + Move parameters to prior
    experiments[
        Description(
            name=f"{dataset_sub_cls}-IB Extract Coreset ({transfer_method}) {bias[1]}",
            seed=seed,
        )
    ] = Experiment(
        dataset=dataset.Generated(
            baseline=baseline.dataset,
            dataset_sub_cls=dataset_sub_cls,
            bias=bias[1],
            convert_to_rgb=("color" in bias[0]),
            shuffle=False,
            valid_size=0.0,
            train_on_reduced_data=False,
            load_coreset=True
        ),
        model=model.ClassificationModelConfig(
            baseline=baseline.model,
            bias=bias[1],
            type="lenet300-100-bayes",
            input_channels=3 if "color" in bias[0] else 1,
            get_intermediate_rep={},
            add_buffer=(),
        ),
        trainer=transfer_config_cls(
            comment=f"{dataset_sub_cls}-IB Update Coreset ({transfer_method}) {bias[1]} ",
            baseline=baseline.trainer,
            extract_coreset={"method": "random", "size": 200},
            reset_for_new_task=True,
            loss_functions={"regression": "CircularDistanceLoss"}
            if "regression" in bias[0]
            else {"img_classification": "CrossEntropyLoss"},
            data_transfer=True,
        ),
        seed=seed,
    )

    # 5: Train on Task B with VCL Loss
    experiments[
        Description(name=f"{dataset_sub_cls}-IB Train VCL {bias[1]}", seed=seed,)
    ] = Experiment(
        dataset=dataset.Generated(
            baseline=baseline.dataset,
            dataset_sub_cls=dataset_sub_cls,
            bias=bias[1],
            convert_to_rgb=("color" in bias[0]),
            train_on_reduced_data=True,
        ),
        model=model.ClassificationModelConfig(
            baseline=baseline.model,
            bias=bias[1],
            input_channels=3 if "color" in bias[0] else 1,
            type="lenet300-100-bayes",
        ),
        trainer=trainer.TrainerConfig(
            comment=f"{dataset_sub_cls}-IB Transfer ({transfer_method}) {bias[1]}",
            baseline=baseline.trainer,
            readout_name="fc3",
            data_transfer=True,
            **trainer_options[transfer_method],
        ),
        seed=seed,
    )

    # 6: Train on Coreset
    experiments[
        Description(
            name=f"{dataset_sub_cls}-IB Train VCL Coreset ({bias[0]},{bias[1]})",
            seed=seed,
        )
    ] = Experiment(
        dataset=dataset.Generated(
            baseline=baseline.dataset,
            dataset_sub_cls=dataset_sub_cls,
            bias=bias[1],
            convert_to_rgb=("color" in bias[0]),
            train_on_coreset=True,
        ),
        model=model.ClassificationModelConfig(
            baseline=baseline.model,
            bias=bias[1],
            input_channels=3 if "color" in bias[0] else 1,
            type="lenet300-100-bayes",
        ),
        trainer=trainer.TrainerConfig(
            comment=f"{dataset_sub_cls}-IB Transfer ({transfer_method}) {bias[1]} -- Coreset",
            baseline=baseline.trainer,
            readout_name="fc3",
            **trainer_options[transfer_method],
        ),
        seed=seed,
    )

    # 7: Test on Task B
    experiments[
        Description(name=f"Test {dataset_sub_cls}-IB {bias[1]}", seed=seed,)
    ] = Experiment(
        dataset=dataset.ImageDatasetConfig(
            comment=f"{dataset_sub_cls}-IB {bias[1]}",
            dataset_cls="MNIST-IB",
            dataset_sub_cls=dataset_sub_cls,
            bias=bias[1],
            convert_to_rgb=("color" in bias[0]),
            baseline=baseline.dataset,
        ),
        model=model.ClassificationModelConfig(
            dataset_cls="MNIST-IB",
            bias=bias[1],
            baseline=baseline.model,
            input_channels=3 if "color" in bias[0] else 1,
            type="lenet300-100-bayes",
        ),
        trainer=trainer.TrainerConfig(
            comment=f"Test {dataset_sub_cls}-IB {bias[1]}",
            baseline=baseline.trainer,
            max_iter=0,
        ),
        seed=seed,
    )

    # 8: Test on Task B
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
            type="lenet300-100-bayes",
        ),
        trainer=trainer.TrainerConfig(
            comment=f"Test {dataset_sub_cls}-IB {bias[2]}",
            baseline=baseline.trainer,
            max_iter=0,
        ),
        seed=seed,
    )

    transfer_experiments[
        Description(
            name=f"Transfer {transfer_method} ({bias[0]}->{bias[1]};{bias[2]})",
            seed=seed,
        )
    ] = TransferExperiment(
        [
            experiments[
                Description(name=f"{dataset_sub_cls}-IB {bias[0]}", seed=seed,)
            ],
            experiments[
                Description(
                    name=f"{dataset_sub_cls}-IB Extract Coreset ({transfer_method}) {bias[0]}",
                    seed=seed,
                )
            ],
            experiments[
                Description(
                    name=f"{dataset_sub_cls}-IB Train VCL {bias[0]}", seed=seed,
                )
            ],
            experiments[
                Description(
                    name=f"{dataset_sub_cls}-IB Extract Coreset ({transfer_method}) {bias[1]}",
                    seed=seed,
                )
            ],
            experiments[
                Description(
                    name=f"{dataset_sub_cls}-IB Train VCL {bias[1]}", seed=seed,
                )
            ],
            experiments[
                Description(
                    name=f"{dataset_sub_cls}-IB Train VCL Coreset ({bias[0]},{bias[1]})",
                    seed=seed,
                )
            ],
            experiments[
                Description(name=f"Test {dataset_sub_cls}-IB {bias[1]}", seed=seed,)
            ],
            experiments[
                Description(name=f"Test {dataset_sub_cls}-IB {bias[2]}", seed=seed,)
            ],
        ]
    )
