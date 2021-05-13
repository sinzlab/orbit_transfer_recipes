import math

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


class DatasetA(ToySineDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.batch_size = 400
        self.size: int = 1000
        self.valid_size = 0.05
        self.sine: dict = {
            "amplitude": (1.0, 1.0),
            "phase": (0.0, 2 * math.pi),
            "freq": (1, 1),
            "x_range": (-5.0, 10.0),
            "samples_per_function": 1,
        }
        super().__init__(**kwargs)


class DatasetB(ToySineDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.batch_size = 50
        self.size: int = 1
        self.sine: dict = {
            "amplitude": (2.0, 2.0),
            "phase": (0, 0),
            "freq": (1, 1),
            "x_range": (-(math.pi/2), math.pi/2),
            "samples_per_function": 50,
        }
        super().__init__(**kwargs)


class DatasetC(DatasetB):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        super().__init__(**kwargs)
        self.sine["samples_per_function"] = 200
        self.sine["x_range"] = (-5.0, 10.0)


class SourceModel(ToySineModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "mlp"
        self.input_size: int = 1
        self.output_size: int = 1
        self.layer_size: int = 10
        self.num_layers: int = 3
        self.activation: str = "sin"
        super().__init__(**kwargs)


class BaselineModel(ToySineModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "mlp"
        self.input_size: int = 1
        self.output_size: int = 1
        self.layer_size: int = 40
        self.num_layers: int = 4
        self.activation: str = "relu"
        super().__init__(**kwargs)

class BaselineTrainer(TransferMixin, ToySineTrainerConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.optimizer: str = "Adam"
        self.optimizer_options = {
            "amsgrad": True,
            "lr": 0.001,
            "weight_decay": 0.001,
        }
        self.max_iter = 1000
        super().__init__(**kwargs)


class GeneratedDatasetB(Generated, DatasetB):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.primary_dataset_fn = "bias_transfer.dataset.toy_sine_dataset_loader"
        super().__init__(**kwargs)


class DataGenerator(DataGenerationMixin, BaselineTrainer):
    fn = "bias_transfer.trainer.toy_regression_transfer"


seed = 14
for transfer in (
    # "L2",
    # "Freeze",
    # "Finetune",
    "FD-MC-Dropout-Cov",
    # "FD-MC-Dropout",
    # "FD",
    # "ELRG L2-SP",
    # "MF L2-SP",
    # "L2-SP",
    # "Diagonal-L2-SP",
    # "EWC",
    # "SynapticIntelligence",
    # "RDL",
    # "KnowledgeDistillation",
):
    gamma = 0.001
    reset = ""
    experiments = []
    softmax_temp = 1.0
    ensemble_members = 40
    rank = 2
    transfer_settings = {
        "L2": [
            {},
            {
                "trainer": {
                    "reset": reset,
                    "optimizer_options": {
                        "amsgrad": False,
                        "lr": 0.001,
                        "weight_decay": gamma,
                    },
                }
            },
        ],
        "FD-MC-Dropout-Cov": [
            {
                "model": {"dropout": 0.2},
            },
            {
                "model": {
                    "get_intermediate_rep": {"layers.6": "layers.6"},
                    "dropout": 0.2,
                },
                "trainer": {
                    "save_representation": True,
                    "save_input": True,
                    "data_transfer": True,
                    "apply_softmax": False,
                    "softmax_temp": 1.0,
                    "compute_covariance": {
                        "type": "full",
                        "precision": "double",
                        "eps": 1e-1,
                        "n_components": ensemble_members,
                        "n_samples": ensemble_members,
                    },
                },
            },
            {
                "model": {
                    "get_intermediate_rep": {"layers.9": "layers.6"},
                    # "add_custom_buffer": {"layers__9_cov_lambdas": (ensemble_members,)},
                    "dropout": 0.01,
                },
                "trainer": {
                    "max_iter": 100,
                    "reset": reset,
                    "single_input_stream": False,
                    "regularization": {
                        "regularizer": "FunctionDistance",
                        "alpha": gamma if gamma != -1 else 1.0,
                        "decay_alpha": False,
                        "softmax_temp": softmax_temp,
                        "use_softmax": False,
                        "cov_eps": 1e-1,
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
        "FD-MC-Dropout": [
            {
                "model": {"dropout": 0.1},
            },
            {
                "model": {
                    "get_intermediate_rep": {"layers.9": "layers.9"},
                    "dropout": 0.1,
                },
                "trainer": {
                    "save_representation": True,
                    "save_input": True,
                    "data_transfer": True,
                    "apply_softmax": False,
                    "softmax_temp": 1.0,
                    "compute_covariance": {
                        "type": "diagonal",
                        "n_samples": ensemble_members,
                    },
                },
            },
            {
                "model": {
                    "get_intermediate_rep": {"layers.9": "layers.9"},
                    "dropout": 0.01,
                },
                "trainer": {
                    "reset": reset,
                    "single_input_stream": False,
                    "regularization": {
                        "regularizer": "FunctionDistance",
                        "alpha": gamma if gamma != -1 else 1.0,
                        "decay_alpha": False,
                        "softmax_temp": softmax_temp,
                        "use_softmax": False,
                        "cov_eps": 1e-12,
                    },
                    "data_transfer": True,
                    "ignore_main_loss": gamma == -1,
                },
            },
            {
                "model": {
                    "dropout": 0.01,
                }
            },
        ],
        "FD": [
            {},
            {
                "model": {
                    "get_intermediate_rep": {"layers.6": "layers.6"},
                },
                "trainer": {
                    "save_representation": True,
                    "save_input": True,
                    "data_transfer": True,
                    "apply_softmax": False,
                    "compute_covariance": {},
                },
            },
            {
                "model": {
                    "get_intermediate_rep": {"layers.6": "layers.6"},
                },
                "trainer": {
                    "reset": reset,
                    "single_input_stream": False,
                    "regularization": {
                        "regularizer": "FunctionDistance",
                        "alpha": gamma if gamma != -1 else 1.0,
                        "decay_alpha": False,
                        "softmax_temp": softmax_temp,
                        "use_softmax": False,
                    },
                    "data_transfer": True,
                    "ignore_main_loss": gamma == -1,
                },
            },
        ],
        "Freeze": [{}, {"trainer": {"reset": reset, "freeze": ("core",)}}],
        "Finetune": [
            {},
            {
                "trainer": {
                    "reset": reset,
                },
            },
        ],
        "Dropout": [
            {},
            {
                "model": {"dropout": gamma},
                "trainer": {
                    "reset": reset,
                },
            },
        ],
        "Mixup": [
            {},
            {
                "trainer": {
                    "reset": reset,
                    "regularization": {
                        "regularizer": "Mixup",
                        "gamma": gamma,
                    },
                }
            },
        ],
        "L2-SP": [
            {},
            {
                "trainer": {
                    "reset": reset,
                    "regularization": {
                        "regularizer": "ParamDistance",
                        "gamma": 1.0,
                        # "ignore_layers": ("fc3",) if "regression" in bias[0] else (),
                    },
                }
            },
        ],
        "Diagonal-L2-SP": [
            {},
            {
                "trainer": {
                    "reset": reset,
                    "regularization": {
                        "regularizer": "ParamDistance",
                        "gamma": gamma,
                        "custom_importance": {
                            "weight": np.array([[1, 0], [0, 1]], dtype=np.float32)
                        },
                    },
                }
            },
        ],
        "MF L2-SP": [
            {
                "model": {
                    "type": "linear-bayes",
                },
                "trainer": {
                    "regularization": {
                        "regularizer": "VCL",
                        "gamma": gamma,
                    },
                },
            },
            {
                "model": {
                    "type": "linear-bayes",
                },
                "trainer": {
                    "bayesian_to_deterministic": True,
                    "reset_for_new_task": True,
                },
            },
            {
                "model": {
                    "add_buffer": ("importance",),
                },
                "trainer": {
                    "reset": reset,
                    "regularization": {
                        "regularizer": "ParamDistance",
                        "gamma": gamma,
                    },
                },
            },
        ],
        "ELRG L2-SP": [
            {
                "model": {
                    "type": "linear-elrg",
                    "alpha": 1 / rank,
                    "rank": rank,
                    "train_var": False,
                    "initial_var": 1e-12,
                },
                "trainer": {
                    "regularization": {
                        "regularizer": "ELRG",
                        "prior_var": 1.0,
                        "num_samples": 100,
                    },
                },
            },
            {
                "model": {
                    "type": "linear-elrg",
                    "alpha": 1 / rank,
                    "rank": rank,
                    "train_var": False,
                    "initial_var": 1e-12,
                },
                "trainer": {
                    "bayesian_to_deterministic": True,
                    "reset_for_new_task": False,
                },
            },
            {
                "model": {
                    "add_buffer": ("importance", ("importance_v", rank)),
                },
                "trainer": {
                    "reset": reset,
                    "regularization": {
                        "regularizer": "ParamDistance",
                        "gamma": gamma,
                        "elrg_alpha": 1 / rank,
                        "use_elrg_importance": True,
                    },
                },
            },
        ],
        "RDL": [
            {},
            {
                "model": {
                    "get_intermediate_rep": {"fc3": "fc3"}
                },  # get_rep["fc2"] = "core" get_rep["conv2"] = "core"
                "trainer": {
                    "save_representation": True,
                    "save_input": True,
                    "data_transfer": True,
                },
            },
            {
                "model": {"get_intermediate_rep": {"fc3": "fc3"}},
                "trainer": {
                    "reset": reset,
                    "single_input_stream": False,
                    "regularization": {
                        "regularizer": "RDL",
                        "gamma": gamma,
                        "dist_measure": "corr",
                        "decay_gamma": False,
                    },
                },
            },
        ],
        "KnowledgeDistillation": [
            {},
            {
                "model": {"get_intermediate_rep": {"fc3": "fc3"}},
                "trainer": {
                    "save_representation": True,
                    "save_input": True,
                    "data_transfer": True,
                },
            },
            {
                "model": {"get_intermediate_rep": {"fc3": "fc3"}},
                "trainer": {
                    "reset": reset,
                    "single_input_stream": False,
                    "regularization": {
                        "regularizer": "KnowledgeDistillation",
                        "gamma": gamma,
                        "decay_gamma": False,
                        "softmax_temp": softmax_temp,
                    },
                },
            },
        ],
        "EWC": [
            {},
            {
                "trainer": {
                    "compute_fisher": {
                        "num_samples": 1024,
                        "empirical": True,
                    }
                }
            },
            {
                "model": {"add_buffer": ("importance",)},
                "trainer": {
                    "reset": reset,
                    "regularization": {
                        "regularizer": "ParamDistance",
                        "gamma": 50000000.0,
                    },
                },
            },
        ],
        "SynapticIntelligence": [
            {"trainer": {"synaptic_intelligence_computation": True}},
            {
                "model": {
                    "add_buffer": ("SI_omega", "SI_prev_task"),
                },
                "trainer": {"compute_si_omega": {"damping_factor": 0.0001}},
            },
            {
                "model": {"add_buffer": ("importance",)},
                "trainer": {
                    "reset": reset,
                    "regularization": {
                        "regularizer": "ParamDistance",
                        "gamma": gamma,
                    },
                },
            },
        ],
    }

    # Step 1: Training on bias[0]
    experiments.append(
        Experiment(
            dataset=DatasetA(),
            model=SourceModel(),
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
                model=SourceModel(),
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
            name=f"{transfer}",
            seed=seed,
        )
    ] = TransferExperiment(experiments, update=transfer_settings[transfer])
