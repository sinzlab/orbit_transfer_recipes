from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

for dataset_cls in (#"CIFAR100",
                    "CIFAR10",
                    ):
    seed = 42
    # Noise augmentation:
    noise_type = {"add_noise": True, "noise_snr": None,
                  "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}}
    noise_adv = {"noise_adv_classification": False, "noise_adv_regression": True}

    experiments[Description(name=dataset_cls + ": Clean", seed=42)] = Experiment(
        dataset=dataset.DatasetConfig(comment=dataset_cls, dataset_cls=dataset_cls, apply_data_normalization=False),
        model=model.ModelConfig(comment=dataset_cls, dataset_cls=dataset_cls),
        trainer=trainer.TrainerConfig(comment=""),
        seed=seed)
    experiments[Description(name=dataset_cls + ": Noise Augmented", seed=42)] = Experiment(
        dataset=dataset.DatasetConfig(comment=dataset_cls, dataset_cls=dataset_cls, apply_data_normalization=False),
        model=model.ModelConfig(comment=dataset_cls, dataset_cls=dataset_cls),
        trainer=trainer.TrainerConfig(comment="Noise Augmented", **noise_type),
        seed=seed)
    matching_options = {"representation": "conv_rep",
                        "criterion": "cosine",
                        "second_noise_std": {(0, 1.0): 1.0},
                        "lambda": 1.0}
    experiments[Description(name=dataset_cls + ": Noise Augmented + Repr. Matching", seed=42)] = Experiment(
        dataset=dataset.DatasetConfig(comment=dataset_cls, dataset_cls=dataset_cls, apply_data_normalization=False),
        model=model.ModelConfig(comment=dataset_cls, dataset_cls=dataset_cls),
        trainer=trainer.TrainerConfig(
            comment="Noise Augmented + Repr. Matching",
            representation_matching=matching_options,
            **noise_type),
        seed=seed)
    matching_options = {"representation": "conv_rep",
                        "criterion": "mse",
                        "second_noise_std": {(0, 0.5): 1.0},
                        "lambda": 1.0}
    experiments[Description(name=dataset_cls + ": Noise Augmented + Repr. Matching (Euclid, (0,0.5))", seed=42)] = Experiment(
        dataset=dataset.DatasetConfig(comment=dataset_cls, dataset_cls=dataset_cls, apply_data_normalization=False),
        model=model.ModelConfig(comment=dataset_cls, dataset_cls=dataset_cls),
        trainer=trainer.TrainerConfig(
            comment="Noise Augmented + Repr. Matching",
            representation_matching=matching_options,
            **noise_type),
        seed=seed)
    experiments[Description(name=dataset_cls + ": Noise Augmented + Noise Adv Regession", seed=42)] = Experiment(
        dataset=dataset.DatasetConfig(comment=dataset_cls, dataset_cls=dataset_cls, apply_data_normalization=False),
        model=model.ModelConfig(comment=dataset_cls, dataset_cls=dataset_cls, noise_adv_regression=True),
        trainer=trainer.TrainerConfig(comment="Noise Augmented + Noise Adv Regression",
                                      noise_adv_loss_factor=1.0,
                                      noise_adv_gamma=10.0,
                                      **noise_adv,
                                      **noise_type),
        seed=seed)

    transfer = Experiment(
        dataset=dataset.DatasetConfig(comment=dataset_cls, dataset_cls=dataset_cls, apply_data_normalization=False),
        model=model.ModelConfig(comment=dataset_cls, dataset_cls=dataset_cls),
        trainer=trainer.TrainerConfig(comment="Transfer + Reset", freeze=("core",), reset_linear=True),
        seed=seed)

    transfer_experiments[Description(name=dataset_cls + ": Clean", seed=42)] = TransferExperiment([
        experiments[Description(name=dataset_cls + ": Clean", seed=42)],
    ])
    transfer_experiments[Description(name=dataset_cls + ": Noise Augmented -> Transfer (Reset)", seed=42)] = TransferExperiment([
        experiments[Description(name=dataset_cls + ": Noise Augmented", seed=42)],
        transfer
    ])
    transfer_experiments[Description(name=dataset_cls + ": Noise Augmented + Noise Adv Regession -> Transfer (Reset)", seed=42)] = TransferExperiment([
        experiments[Description(name=dataset_cls + ": Noise Augmented + Noise Adv Regession", seed=42)],
        transfer
    ])
    transfer_experiments[Description(name=dataset_cls + ": Noise Augmented + Repr. Matching (Euclid, (0,0.5)) -> Transfer (Reset)",
                                     seed=42)] = TransferExperiment([
        experiments[Description(name=dataset_cls + ": Noise Augmented + Repr. Matching (Euclid, (0,0.5))", seed=42)],
        transfer
    ])
    transfer_experiments[Description(name=dataset_cls + ": Noise Augmented + Repr. Matching -> Transfer (Reset)", seed=42)] = TransferExperiment([
        experiments[Description(name=dataset_cls + ": Noise Augmented + Repr. Matching", seed=42)],
        transfer
    ])
