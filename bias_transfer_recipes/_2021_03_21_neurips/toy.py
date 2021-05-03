import numpy as np

from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from bias_transfer.configs import *

transfer_experiments = {}


class DatasetA(ToyDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.cov_a = [[40, 0], [0, 0.001]]
        self.cov_b = [[40, 0], [0, 0.001]]
        self.mu_a = [0, 0.5]
        self.mu_b = [0, -0.5]
        super().__init__(**kwargs)


class DatasetB(ToyDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.cov_a = [[0.05, 0], [0, 0.001]]
        self.cov_b = [[0.05, 0], [0, 0.001]]
        self.mu_a = [0.5, -0.5]
        self.mu_b = [-0.5, 0.5]
        super().__init__(**kwargs)


class DatasetC(ToyDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.cov_a = [[40, 0], [0, 0.001]]
        self.cov_b = [[40, 0], [0, 0.001]]
        self.mu_a = [0, -0.5]
        self.mu_b = [0, 0.5]
        super().__init__(**kwargs)


class BaselineModel(ToyModel):
    pass


class BaselineTrainer(TransferMixin, ToyTrainerConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.optimizer: str = "Adam"
        self.optimizer_options = {
            "amsgrad": True,
            "lr": 0.1,
            "weight_decay": 0.001,
        }
        self.max_iter = 1000
        super().__init__(**kwargs)


class DataGenerator(DataGenerationMixin, ToyTrainerConfig):
    fn = "bias_transfer.trainer.transfer"


seed = 6
for transfer in (
    # "L2",
    # "Freeze",
    "Finetune",
    "ELRG L2-SP",
    # "MF L2-SP",
    "L2-SP",
    # "Diagonal-L2-SP",
    "EWC",
    "SynapticIntelligence",
    # "RDL",
    # "KnowledgeDistillation",
):
    gamma = 1.0
    reset = "all"
    experiments = []
    softmax_temp = 1.0
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
                    "initial_var": 1e-12
                },
                "trainer": {
                    "regularization": {
                        "regularizer": "ELRG",
                        "prior_var": 1.0,
                        "num_samples": 100
                    },
                },
            },
            {
                "model": {
                    "type": "linear-elrg",
                    "alpha": 1 / rank,
                    "rank": rank,
                    "train_var": False,
                    "initial_var": 1e-12
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
        "MF L2-SP"
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

    # Step 2: Training on bias[1]
    experiments.append(
        Experiment(
            dataset=DatasetB(),
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
