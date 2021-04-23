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
        self.cov_a = [[20, 0], [0, 0.01]]
        self.cov_b = [[20, 0], [0, 0.01]]
        self.mu_a = [4, 1]
        self.mu_b = [4, -1]
        super().__init__(**kwargs)


class DatasetB(ToyDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.cov_a = [[0.01, 0], [0, 0.01]]
        self.cov_b = [[0.01, 0], [0, 0.01]]
        self.mu_a = [0, -1]
        self.mu_b = [-3, 1]
        super().__init__(**kwargs)


class DatasetC(ToyDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.cov_a = [[20, 0], [0, 0.01]]
        self.cov_b = [[20, 0], [0, 0.01]]
        self.mu_a = [-4, -1]
        self.mu_b = [-4, 1]
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
        }
        self.max_iter = 1000
        super().__init__(**kwargs)


class DataGenerator(DataGenerationMixin, ToyTrainerConfig):
    fn = "bias_transfer.trainer.transfer"


seed = 42
for transfer in (
    # "L2",
    # "Freeze",
    "Finetune",
    # "L2-SP",
    # "Diagonal-L2-SP",
    # "EWC",
    # "SynapticIntelligence",
    # "RDL",
    # "KnowledgeDistillation",
):
    alpha = 0.01
    reset = "all"
    experiments = []
    softmax_temp = 1.0
    transfer_settings = {
        "L2": [
            {},
            {
                "trainer": {
                    "reset": reset,
                    "optimizer_options": {
                        "amsgrad": False,
                        "lr": 0.001,
                        "weight_decay": alpha,
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
                "model": {"dropout": alpha},
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
                        "alpha": alpha,
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
                        "alpha": alpha,
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
                        "alpha": alpha,
                        "custom_importance": {
                            "weight": np.array(
                                [[1, 0], [0, 1]], dtype=np.float32
                            )
                        },
                    },
                }
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
                        "alpha": alpha,
                        "dist_measure": "corr",
                        "decay_alpha": False,
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
                        "alpha": alpha,
                        "decay_alpha": False,
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
                        "alpha": alpha,
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
                        "alpha": alpha,
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
            name=f"{transfer} {reset_string}: {alpha}",
            seed=seed,
        )
    ] = TransferExperiment(experiments, update=transfer_settings[transfer])
