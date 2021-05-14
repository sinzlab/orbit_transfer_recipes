import math
from itertools import product

import numpy as np

from bias_transfer.configs.dataset.toy import ToySineDatasetConfig
from bias_transfer.configs.model.toy import ToySineModel
from bias_transfer.configs.trainer.toy import ToySineTrainerConfig
from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from bias_transfer.configs import *

transfer_experiments = {}


class DatasetB(ToySineDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.batch_size = 50
        self.size: int = 1
        self.sine: dict = {
            "amplitude": (2.0, 2.0),
            "phase": (0, 0),
            "freq": (1, 1),
            "x_range": (-(math.pi / 2), math.pi / 2),
            "samples_per_function": 50,
        }
        super().__init__(**kwargs)


class DatasetC(DatasetB):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        super().__init__(**kwargs)
        self.sine["samples_per_function"] = 200
        self.sine["x_range"] = (-5.0, 10.0)


class BaselineTrainer(TransferMixin, ToySineTrainerConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.optimizer: str = "Adam"
        self.optimizer_options = {
            "amsgrad": True,
            "lr": 0.001,
            "weight_decay": 0.001,
        }
        self.max_iter = 2000
        super().__init__(**kwargs)


class GeneratedDatasetB(Generated, DatasetB):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.primary_dataset_fn = "bias_transfer.dataset.toy_sine_dataset_loader"
        super().__init__(**kwargs)


class DataGenerator(DataGenerationMixin, BaselineTrainer):
    fn = "bias_transfer.trainer.toy_regression_transfer"


seed = 42
transfer = "FD-MC-Dropout-Cov"
for (
    gamma,
    dropout,
    ensemble_members,
    eps,
    amplitude_delta,
    phase_random,
    num_layers,
    layer_width,
    size_ratio,
) in product(
    (
        0.001,
        # 0.01,
    ),  # gamma
    (
               0.1,
        0.2,
        0.3,
        0.5,
    ),  # dropout
    (
        40,
        # 100,
    ),  # ensemble members
    (1e-1,
     1e-5,
     1e-10
     ),  # eps
    (
        # 0.05,
        # 0.01,
            0.0,
        0.1,
        0.3,
        0.5,
        1.0,
        2.0,
    ),  # amplitude delta
    ( 0.0,
        0.1,
        0.3,
        0.5,
        1.0,
        math.pi,
    ),  # phase delta
    (
        #3,
        4,
    ),  # num layers
    (80,),  # 40,  # hidden size
    (
        # (2000, 1, 400),
        # (200, 40, 10),
        # (500, 40, 10),
        (40, 400, 200),
        (100, 100, 200),
        (300, 50, 200),
    ),  # (size, samples per fct, batch-size)
):
    phase_delta = amplitude_delta if phase_random else 0.0

    class DatasetA(ToySineDatasetConfig):
        def __init__(self, **kwargs):
            self.load_kwargs(**kwargs)
            self.batch_size = size_ratio[2]
            self.size: int = size_ratio[0]
            self.valid_size = 0.05
            self.sine: dict = {
                "amplitude": (1.0 - amplitude_delta, 1.0 + amplitude_delta),
                "phase": (0.0 - phase_delta, 0.0 + phase_delta),
                "freq": (1, 1),
                "x_range": (-5.0, 10.0),
                "samples_per_function": size_ratio[1],
                "multi_regression": True,
            }
            super().__init__(**kwargs)

    class BaselineModel(ToySineModel):
        def __init__(self, **kwargs):
            self.load_kwargs(**kwargs)
            self.type: str = "mlp"
            self.input_size: int = 1
            self.output_size: int = 1
            self.layer_size: int = layer_width
            self.num_layers: int = num_layers
            self.activation: str = "relu"
            super().__init__(**kwargs)

    last_layer = "layers.6" if num_layers == 4 else "layers.9"
    reset = ""
    experiments = []
    transfer_settings = {
        "FD-MC-Dropout-Cov": [
            {
                "model": {"dropout": dropout,
                          "output_size": size_ratio[0],
                          },
            },
            {
                "model": {
                    "get_intermediate_rep": {last_layer: last_layer},
                    "dropout": dropout,
                    "output_size": size_ratio[0],
                },
                "trainer": {
                    "save_representation": True,
                    "save_input": True,
                    "data_transfer": True,
                    "apply_softmax": False,
                    "compute_covariance": {
                        "type": "full",
                        "precision": "double",
                        "eps": eps,
                        "n_components": ensemble_members,
                        "n_samples": ensemble_members,
                    },
                },
            },
            {
                "model": {
                    "get_intermediate_rep": {last_layer: last_layer},
                    # "add_custom_buffer": {"layers__9_cov_lambdas": (ensemble_members,)},
                    "dropout": 0.01,
                },
                "trainer": {
                    "max_iter": 300,
                    "reset": reset,
                    "single_input_stream": False,
                    "regularization": {
                        "regularizer": "FunctionDistance",
                        "alpha": gamma if gamma != -1 else 1.0,
                        "decay_alpha": False,
                        "use_softmax": False,
                        "cov_eps": eps,
                    },
                    "data_transfer": True,
                    "ignore_main_loss": gamma == -1,
                    "optim_step_count": 2,
                },
            },
            {
                "model": {
                    "dropout": 0.01,
                }
            },
        ],
    }

    # Step 1: Training on bias[0]
    experiments.append(
        Experiment(
            dataset=DatasetA(),
            model=BaselineModel(),
            trainer=BaselineTrainer(),
            seed=seed,
        )
    )

    # (Step 1.1: Data Generation)
    if transfer in (
        "RDL",
        "KnowledgeDistillation",
        "EWC",
        "SynapticIntelligence",
        "ELRG L2-SP",
        "MF L2-SP",
        "FD",
        "FD-MC-Dropout",
        "FD-MC-Dropout-Cov",
    ):
        experiments.append(
            Experiment(
                dataset=DatasetA(
                    shuffle=False,
                    valid_size=0.0,
                ),
                model=BaselineModel(),
                trainer=DataGenerator(comment=f"Data Generation ({transfer})"),
                seed=seed,
            )
        )

    if transfer in (
        "RDL",
        "KnowledgeDistillation",
        "FROMP",
        "FD",
        "FD-MC-Dropout",
        "FD-MC-Dropout-Cov",
    ):
        target_dataset = GeneratedDatasetB
    else:
        target_dataset = DatasetB

    # Step 2: Training on bias[1]
    experiments.append(
        Experiment(
            dataset=target_dataset(),
            model=BaselineModel(),
            trainer=BaselineTrainer(),
            seed=seed,
        )
    )

    # Step 3: Test on bias[2]
    experiments.append(
        Experiment(
            dataset=DatasetC(),
            model=BaselineModel(),
            trainer=BaselineTrainer(max_iter=0),
            seed=seed,
        )
    )

    reset_string = "reset" if reset == "all" else ""
    transfer_experiments[
        Description(
            name=f"{transfer}"
            + str(
                (
                    gamma,
                    dropout,
                    ensemble_members,
                    eps,
                    amplitude_delta,
                    phase_random,
                    num_layers,
                    layer_width,
                    size_ratio,
                )
            ),
            seed=seed,
        )
    ] = TransferExperiment(experiments, update=transfer_settings[transfer])
