from itertools import product

from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from bias_transfer.configs import *

transfer_experiments = {}


class BaselineDataset(MNISTTransfer):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_sub_cls = "FashionMNIST"
        super().__init__(**kwargs)


class GeneratedDataset(Generated, BaselineDataset):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        super().__init__(**kwargs)


class BaselineModel(MNISTTransferModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "lenet300-100"
        self.coreset_size = 200
        super().__init__(**kwargs)


class BaselineTrainer(TransferMixin, Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.patience = 1000
        super().__init__(**kwargs)


class BaselineSimclrTrainer(SimclrMixin, BaselineTrainer):
    pass


class BaselineRegressionTrainer(TransferMixin, Regression):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.patience = 1000
        self.readout_name = "fc3"
        self.loss_functions = {"regression": "MSELoss"}
        super().__init__(**kwargs)


class DataGenerator(DataGenerationMixin, Classification):
    fn = "bias_transfer.trainer.transfer"


class DataGeneratorSimclr(SimclrMixin, DataGenerationMixin, Classification):
    fn = "bias_transfer.trainer.transfer"


class DataGeneratorRegression(DataGenerationMixin, Regression):
    fn = "bias_transfer.trainer.regression_transfer"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.loss_functions = {"regression": "CircularDistanceLoss"}
        super(DataGeneratorRegression, self).__init__(**kwargs)


possible_settings = {
    "Finetune": ((1.0,), (-1,), (-1,)),  # alpha
    "L2-SP": ((1.0, 0.1, 10.0, 5.0), (-1,), (-1,)),  # alpha
    "EWC": ((1.0, 0.1, 10.0, 5.0), (-1,), (-1,)),  # alpha
    "SynapticIntelligence": ((1.0, 0.1, 10.0, 5.0), (-1,), (-1,)),  # alpha
    "ELRG L2-SP": (
        (1.0, 0.1, 10.0, 0.01),
        (1, 5, 10, 20),
        (1e-1, 1e-5, 1e-12),
    ),  # alpha, rank, eps
    "MF L2-SP": ((1.0,), (-1,), (-1,)),  # alpha
}

seed = 42
for environment in (
    (
        ("clean", "classification", "fc"),
        ("color", "classification", "fc"),
        ("color_shuffle", "classification", "fc"),
    ),
    # (
    #     ("noise", "simclr", "conv"),
    #     ("low_resource", "classification", "conv"),
    #     ("noise", "classification", "conv"),
    # ),
    # (
    #     ("clean", "classification", "conv"),
    #     ("clean", "classification", "mlp"),
    #     ("translation", "classification", "mlp"),
    # ),
    (
        ("scale", "split-classification 0-4", "fc"),
        ("clean", "split-classification 5-9", "fc"),
        ("scale", "classification", "fc"),
    ),
):
    for transfer in (
        "Finetune",
        "L2-SP",
        "EWC",
        "SynapticIntelligence",
        "ELRG L2-SP",
        # "MF L2-SP",
        # "VCL",
    ):
        for settings in product(*possible_settings[transfer]):
            experiments = []
            softmax_temp = 1.0
            transfer_settings = {
                "Finetune": [],
                "L2-SP": [
                    {},
                    {
                        "trainer": {
                            "regularization": {
                                "regularizer": "ParamDistance",
                                "gamma": 1.0,
                                # "ignore_layers": ("fc3",) if "regression" in bias[0] else (),
                            },
                        }
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
                            "regularization": {
                                "regularizer": "ParamDistance",
                                "gamma": settings[0],
                            },
                        },
                    },
                ],
                "VCL": [
                    {},
                    {
                        "trainer": {
                            "regularization": {
                                "regularizer": "VCL",
                            },
                        },
                    },
                    {
                        "trainer": {
                            "reset_for_new_task": True,
                        },
                    },
                    {
                        "trainer": {
                            "regularization": {
                                "regularizer": "VCL",
                            },
                        },
                    },
                ],
                "MF L2-SP": [
                    {
                        "trainer": {
                            "regularization": {
                                "regularizer": "VCL",
                            },
                        },
                    },
                    {
                        "trainer": {
                            "bayesian_to_deterministic": True,
                            "reset_for_new_task": True,
                        },
                    },
                    {
                        "model": {
                            "type": "lenet300-100",
                            "add_buffer": ("importance",),
                        },
                        "trainer": {
                            "regularization": {
                                "regularizer": "ParamDistance",
                                "gamma": settings[0],
                                "ignore_layers": ("fc3",)
                                if "regression" in environment[0][1]
                                or "simclr" in environment[0][1]
                                else (),
                            },
                        },
                    },
                    {
                        "model": {"type": "lenet300-100"},
                    },
                ],
                "ELRG L2-SP": [
                    {
                        "model": {
                            "type": "lenet300-100-elrg",
                            "alpha": 1 / settings[1],
                            "rank": settings[1],
                            "train_var": False,
                            "initial_var": settings[2],
                        },
                        "trainer": {
                            "regularization": {
                                "regularizer": "ELRG",
                                "prior_var": 1.0,
                                "num_samples": 10,
                            },
                        },
                    },
                    {
                        "model": {
                            "type": "lenet300-100-elrg",
                            "alpha": 1 / settings[1],
                            "rank": settings[1],
                            "train_var": False,
                            "initial_var": settings[2],
                        },
                        "trainer": {
                            "bayesian_to_deterministic": True,
                            "reset_for_new_task": False,
                        },
                    },
                    {
                        "model": {
                            "add_buffer": ("importance", ("importance_v", settings[1])),
                        },
                        "trainer": {
                            "regularization": {
                                "regularizer": "ParamDistance",
                                "gamma": settings[0],
                                "use_elrg_importance": True,
                                "ignore_layers": ("fc3",)
                                if "regression" in environment[0][1]
                                or "simclr" in environment[0][1]
                                else (),
                            },
                        },
                    },
                ],
            }
            if environment[0][1] == "simclr":
                trainer_config_cls = BaselineSimclrTrainer
                transfer_config_cls = DataGeneratorSimclr
            elif environment[0][1] == "regression":
                trainer_config_cls = BaselineRegressionTrainer
                transfer_config_cls = DataGeneratorRegression
            else:
                trainer_config_cls = BaselineTrainer
                transfer_config_cls = DataGenerator

            if "split" in environment[0][1]:
                split = tuple(map(int, environment[0][1].split()[1].split("-")))
            else:
                split = ()

            # Step 1: Training on source_bias
            experiments.append(
                Experiment(
                    dataset=BaselineDataset(
                        bias=environment[0][0],
                        convert_to_rgb=("color" in environment[1][0]),
                        filter_classes=split,
                        reduce_to_filtered_classes=False,
                    ),
                    model=BaselineModel(
                        bias=environment[0][0],
                        input_channels=3 if "color" in environment[1][0] else 1,
                        get_intermediate_rep={"fc2": "fc2"}
                        if environment[0][1] == "simclr"
                        else {},
                    ),
                    trainer=trainer_config_cls(
                        comment=f"MNIST-Transfer {environment[0][0]}",
                    ),
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
            ):
                # Step 1.1: Data Generation
                experiments.append(
                    Experiment(
                        dataset=BaselineDataset(
                            bias=environment[0][0],
                            convert_to_rgb=("color" in environment[1][0]),
                            shuffle=False,
                            valid_size=0.0,
                            filter_classes=split,
                            reduce_to_filtered_classes=False,
                        ),
                        model=BaselineModel(
                            bias=environment[0][0],
                            input_channels=3 if "color" in environment[1][0] else 1,
                            get_intermediate_rep={"fc2": "fc2"}
                            if environment[1][1] == "simclr"
                            else {},
                        ),
                        trainer=transfer_config_cls(
                            comment=f"MNIST Data Generation ({transfer}) {environment[1][0]}",
                        ),
                        seed=seed,
                    )
                )

            # Step 2: Training on target environment
            if "split" in environment[1][1]:
                split = tuple(map(int, environment[1][1].split()[1].split("-")))
            else:
                split = ()

            experiments.append(
                Experiment(
                    dataset=BaselineDataset(
                        dataset_cls="MNIST-Transfer",
                        bias=environment[1][0],
                        filter_classes=split,
                        reduce_to_filtered_classes=False,
                    ),
                    model=BaselineModel(
                        bias=environment[1][0],
                    ),
                    trainer=BaselineTrainer(
                        comment=f"MNIST Transfer ({transfer}) {environment[1][0]}",
                    ),
                    seed=seed,
                )
            )

            # Step 3: Test on bias[2]
            experiments.append(
                Experiment(
                    dataset=BaselineDataset(
                        bias=environment[2][0],
                    ),
                    model=BaselineModel(
                        bias=environment[2][0],
                        input_channels=3 if "color" in environment[1][0] else 1,
                    ),
                    trainer=BaselineTrainer(
                        comment=f"Test MNIST-Transfer {environment[2][0]}",
                        max_iter=0,
                    ),
                    seed=seed,
                )
            )

            transfer_experiments[
                Description(
                    name=f"{transfer} ::: {settings} ::: ({environment[0][0]}->{environment[1][0]};{environment[2][0]})",
                    seed=seed,
                )
            ] = TransferExperiment(experiments, update=transfer_settings[transfer])
