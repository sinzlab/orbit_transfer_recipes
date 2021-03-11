from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

transfer_experiments = {}


class BaselineDataset(dataset.MNIST_IB):
    pass


class BaselineModel(model.MNISTIB):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.coreset_size = 200
        super(BaselineModel, self).__init__(**kwargs)


class BaselineTrainer(trainer.mixins.TransferMixin, trainer.Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.patience = 1000
        super(BaselineTrainer, self).__init__(**kwargs)


class BaselineRegressionTrainer(trainer.mixins.TransferMixin, trainer.Regression):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.patience = 1000
        self.readout_name = "fc3"
        self.loss_functions = {"regression": "CircularDistanceLoss"}
        super(BaselineRegressionTrainer, self).__init__(**kwargs)


class DataGenerator(trainer.mixins.DataGenerationMixin, trainer.Classification):
    fn = "bias_transfer.trainer.transfer"


class DataGeneratorRegression(trainer.mixins.DataGenerationMixin, trainer.Regression):
    fn = "bias_transfer.trainer.regression_transfer"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.loss_functions = {"regression": "CircularDistanceLoss"}
        super(DataGeneratorRegression, self).__init__(**kwargs)


seed = 42
for bias in (
    ("clean", "color", "color_shuffle"),
    ("noise", "clean", "noise"),
    ("translation", "clean", "translation"),
    # ("rotation_regression", "clean", "rotation"),
):
    for transfer, alphas, resets in (
        # ("L2", (0.0001, 0.001, 0.01, 0.1, 0.005, 0.0005), ("",)),
        # ("Mixup", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), ("",)),
        ("Freeze", ("",), ("",)),
        ("Finetune", ("",), ("",)),
        # ("Dropout", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), ("",)),
        ("L2-SP", (0.1, 0.5, 1.0, 2.0, 5.0, 10.0), ("",
                                                    # "all"
                                                    )),
        ("EWC", (0.1, 0.5, 1.0, 2.0, 5.0, 10.0), ("",
                                                  # "all"
                                                  )),
        ("SynapticIntelligence", (0.1, 0.5, 1.0, 2.0, 5.0, 10.0), ("",
                                                                   # "all"
                                                                   )),
        ("RDL", (0.1, 0.5, 1.0, 2.0, 5.0, 10.0), ("",
                                                  # "all"
                                                  )),
        ("KnowledgeDistillation", (0.1, 0.5, 1.0, 2.0, 5.0, 10.0), ("",
                                                                    # "all"
                                                                    )),
    ):
        for alpha in alphas:
            for reset in resets:
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
                                    "lr": 0.0003,
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
                                    "ignore_layers": ("fc3",)
                                    if "regression" in bias[0]
                                    else (),
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

                if transfer == "KnowledgeDistillation" and "regression" in bias[0]:
                    continue
                trainer_config_cls = (
                    trainer.Regression if "regression" in bias[0] else BaselineTrainer
                )
                transfer_config_cls = (
                    DataGeneratorRegression
                    if "regression" in bias[0]
                    else DataGenerator
                )

                # Step 1: Training on bias[0]
                experiments.append(
                    Experiment(
                        dataset=BaselineDataset(
                            bias=bias[0],
                            convert_to_rgb=("color" in bias[1]),
                        ),
                        model=BaselineModel(
                            bias=bias[0],
                            input_channels=3 if "color" in bias[1] else 1,
                            type="lenet300-100"
                            if bias[0] == "translation"
                            else "lenet5",
                        ),
                        trainer=trainer_config_cls(
                            comment=f"FashionMNIST-IB {bias[0]}",
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
                ):
                    experiments.append(
                        Experiment(
                            dataset=BaselineDataset(
                                bias=bias[0],
                                shuffle=False,
                                valid_size=0.0,
                                convert_to_rgb=("color" in bias[1]),
                            ),
                            model=BaselineModel(
                                bias=bias[0],
                                input_channels=3 if "color" in bias[1] else 1,
                                type="lenet300-100"
                                if bias[0] == "translation"
                                else "lenet5",
                            ),
                            trainer=transfer_config_cls(
                                comment=f" FashionMNIST Data Generation ({transfer}) {bias[0]}",
                            ),
                            seed=seed,
                        )
                    )

                # Step 2: Training on bias[1]
                experiments.append(
                    Experiment(
                        dataset=BaselineDataset(
                            dataset_cls="MNIST-IB",
                            bias=bias[1],
                        ),
                        model=BaselineModel(
                            bias=bias[1],
                            type="lenet300-100"
                            if bias[0] == "translation"
                            else "lenet5",
                        ),
                        trainer=BaselineTrainer(
                            comment=f"FashionMNIST Transfer ({transfer}) {bias[1]}",
                        ),
                        seed=seed,
                    )
                )

                # Step 3: Test on bias[2]
                experiments.append(
                    Experiment(
                        dataset=BaselineDataset(
                            bias=bias[2],
                        ),
                        model=BaselineModel(
                            bias=bias[2],
                            input_channels=3 if "color" in bias[1] else 1,
                            type="lenet300-100"
                            if bias[0] == "translation"
                            else "lenet5",
                        ),
                        trainer=BaselineTrainer(
                            comment=f"Test FashionMNIST-IB {bias[2]}",
                            max_iter=0,
                        ),
                        seed=seed,
                    )
                )

                reset_string = "reset" if reset == "all" else ""
                transfer_experiments[
                    Description(
                        name=f"{transfer} {reset_string}: {alpha} ({bias[0]}->{bias[1]};{bias[2]})",
                        seed=seed,
                    )
                ] = TransferExperiment(experiments, update=transfer_settings[transfer])
