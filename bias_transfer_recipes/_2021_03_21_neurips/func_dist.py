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
        self.batch_size = 64
        self.dataset_sub_cls = "FashionMNIST"
        super().__init__(**kwargs)


class BaselineModel(MNISTTransferModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.coreset_size = 50
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


class GeneratedDataset(Generated, BaselineDataset):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.primary_dataset_fn = "bias_transfer.dataset.mnist_transfer_dataset_loader"
        super().__init__(**kwargs)


class DataGeneratorSimclr(SimclrMixin, DataGenerationMixin, Classification):
    fn = "bias_transfer.trainer.transfer"


class DataGeneratorRegression(DataGenerationMixin, Regression):
    fn = "bias_transfer.trainer.regression_transfer"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.loss_functions = {"regression": "CircularDistanceLoss"}
        super(DataGeneratorRegression, self).__init__(**kwargs)


possible_settings = {
    "Finetune": ((1.0,), (None,), (None,), (None,), (None,)),  # alpha
    "FROMP": (
        (0.01, 0.1, 1.0, 2.0, 10.0),
        (None,),
        (None,),
        (None,),
        (None,),
    ),  # alpha
    "KnowledgeDistillation": (
        (-1, 0.01, 0.1, 1.0, 2.0, 10.0),  # alpha
        (1.0, 10.0, 0.1, 100.0),  # softmax temp
        (None,),
        (None,),
        (None,),
    ),
    "FD": (
        (0.001, 0.01, 0.1, 1.0),  # alpha
        (True, False),  # use softmax
        (None,),
        (None,),
        (None,),
    ),
    "FD-MC-Dropout": (
        (0.1, 1.0),  # alpha
        (True, False),  # use softmax
        (0.1, 0.3, 0.5),  # dropout
        (10, 100),  # ensemble members
        (1e-10,),  # eps
    ),
    "FD-MC-Dropout-Cov": (
        (0.001, 0.1),  # alpha
        (True, False),  # use softmax
        (0.1, 0.3, 0.5),  # dropout
        (10, 100),  # ensemble members
        (1e-1, 1e-5, 1e-10),  # eps
    ),
}

seed = 8
for environment in (
    # (
    #     ("clean", "classification", "conv"),
    #     ("color", "classification", "conv"),
    #     ("color_shuffle", "classification", "conv"),
    # ),
    # (
    #     ("noise", "simclr", "conv"),
    #     ("low_resource", "classification", "conv"),
    #     ("noise", "classification", "conv"),
    # ),
    (
        ("translation_positive", "classification", "conv"),
        ("clean", "classification", "lc"),
        ("translation", "classification", "lc"),
    ),
    (
        ("translation_positive", "classification", "conv"),
        ("clean", "classification", "fc"),
        ("translation", "classification", "fc"),
    ),
    # (
    #         ("translation", "classification", "conv"),
    #         ("clean", "classification", "lc"),
    #         ("translation", "classification", "lc"),
    # ),
    # (
    #         ("translation", "classification", "conv"),
    #         ("clean", "classification", "fc"),
    #         ("translation", "classification", "fc"),
    # ),
    # (
    #     ("clean", "classification", "conv"),
    #     ("clean", "classification", "lc"),
    #     ("translation", "classification", "lc"),
    # ),
    # (
    #     ("clean", "classification", "conv"),
    #     ("clean", "classification", "fc"),
    #     ("translation", "classification", "fc"),
    # ),
    # (
    #     ("scale", "split-classification 0-4", "conv"),
    #     ("clean", "split-classification 5-9", "conv"),
    #     ("scale", "classification", "conv"),
    # ),
):
    for transfer in (
        "Finetune",
        "FROMP",
        "FD",
        "FD-MC-Dropout",
        "FD-MC-Dropout-Cov",
        "KnowledgeDistillation",
    ):
        for settings in product(*possible_settings[transfer]):
            alpha = settings[0]
            experiments = []
            transfer_settings = {
                "Finetune": [],
                "FD-MC-Dropout-Cov": [
                    {
                        "model": {"dropout": settings[2]},
                    },
                    {
                        "model": {
                            "get_intermediate_rep": {"fc3": "fc3"},
                            "dropout": settings[2],
                        },
                        "trainer": {
                            "save_representation": True,
                            "save_input": True,
                            "data_transfer": True,
                            "apply_softmax": settings[1],
                            "softmax_temp": 1.0,
                            "compute_covariance": {
                                "type": "full",
                                "precision": "double",
                                "eps": settings[4],
                                "n_components": settings[3],
                                "n_samples": settings[3],
                            },
                        },
                    },
                    {
                        "model": {
                            "get_intermediate_rep": {"fc3": "fc3"},
                            # "add_custom_buffer": {"fc3_cov_lambdas": (10,)},
                        },
                        "trainer": {
                            "reset": "all",
                            "single_input_stream": False,
                            "regularization": {
                                "regularizer": "FunctionDistance",
                                "alpha": alpha if alpha != -1 else 1.0,
                                "decay_alpha": False,
                                "softmax_temp": 1.0,
                                "use_softmax": settings[1],
                                "cov_eps": settings[4],
                            },
                            "data_transfer": True,
                            "ignore_main_loss": alpha == -1,
                            "optim_step_count": 2 if alpha != -1 else 1,
                        },
                    },
                ],
                "FD-MC-Dropout": [
                    {
                        "model": {"dropout": settings[2]},
                    },
                    {
                        "model": {
                            "get_intermediate_rep": {"fc3": "fc3"},
                            "dropout": settings[2],
                        },
                        "trainer": {
                            "save_representation": True,
                            "save_input": True,
                            "data_transfer": True,
                            "apply_softmax": settings[1],
                            "softmax_temp": 1.0,
                            "compute_covariance": {
                                "type": "diagonal",
                                "eps": settings[4],
                                "n_components": settings[3],
                                "n_samples": settings[3],
                            },
                        },
                    },
                    {
                        "model": {"get_intermediate_rep": {"fc3": "fc3"}},
                        "trainer": {
                            "reset": "all",
                            "single_input_stream": False,
                            "regularization": {
                                "regularizer": "FunctionDistance",
                                "alpha": alpha if alpha != -1 else 1.0,
                                "decay_alpha": False,
                                "softmax_temp": 1.0,
                                "use_softmax": settings[1],
                                "cov_eps": settings[4],
                            },
                            "data_transfer": True,
                            "ignore_main_loss": alpha == -1,
                            "optim_step_count": 2 if alpha != -1 else 1,
                        },
                    },
                ],
                "FD": [
                    {},
                    {
                        "model": {
                            "get_intermediate_rep": {"fc3": "fc3"},
                        },
                        "trainer": {
                            "compute_covariance": {},
                            "save_representation": True,
                            "save_input": True,
                            "data_transfer": True,
                            "apply_softmax": settings[1],
                            "softmax_temp": 1.0,
                        },
                    },
                    {
                        "model": {"get_intermediate_rep": {"fc3": "fc3"}},
                        "trainer": {
                            "reset": "all",
                            "single_input_stream": False,
                            "regularization": {
                                "regularizer": "FunctionDistance",
                                "alpha": alpha if alpha != -1 else 1.0,
                                "decay_alpha": False,
                                "softmax_temp": 1.0,
                                "use_softmax": settings[1],
                            },
                            "data_transfer": True,
                            "ignore_main_loss": alpha == -1,
                            "optim_step_count": 2 if alpha != -1 else 1,
                        },
                    },
                ],
                "KnowledgeDistillation": [
                    {},
                    {
                        "model": {"get_intermediate_rep": {"fc3": "fc3"}},
                        "trainer": {
                            "compute_covariance": {},
                            "save_representation": True,
                            "save_input": True,
                            "data_transfer": True,
                        },
                    },
                    {
                        "model": {"get_intermediate_rep": {"fc3": "fc3"}},
                        "trainer": {
                            "reset": "all",
                            "single_input_stream": False,
                            "regularization": {
                                "regularizer": "KnowledgeDistillation",
                                "alpha": alpha if alpha != -1 else 1.0,
                                "decay_alpha": False,
                                "softmax_temp": settings[1],
                            },
                            "data_transfer": True,
                            "ignore_main_loss": alpha == -1,
                            "optim_step_count": 2 if alpha != -1 else 1,
                        },
                    },
                ],
                "FROMP": [
                    {},
                    {
                        "trainer": {
                            "data_transfer": True,
                            "extract_coreset": {
                                "method": "fromp",
                                "size": 50,
                            },
                            "compute_covariance": {"batch_size": 32},
                            "regularization": {
                                "regularizer": "FROMP",
                                "prior_prec": 1e-4,
                                "eps": 1e-8,
                                "grad_clip_norm": 100,
                                "alpha": alpha,
                            },
                            "init_fromp": True,
                        },
                    },
                    {
                        "dataset": {"load_coreset": True},
                        "trainer": {
                            "reset": "all",
                            "regularization": {
                                "regularizer": "FROMP",
                                "prior_prec": 1e-4,
                                "eps": 1e-8,
                                "grad_clip_norm": 100,
                                "alpha": alpha,
                            },
                            "data_transfer": True,
                        },
                    },
                    {},
                ],
            }

            if transfer == "KnowledgeDistillation" and (
                "regression" in environment[0][1] or "simclr" in environment[0][1]
            ):
                continue
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
                        type="lenet5",
                        core_type=environment[0][2],
                        get_intermediate_rep={"fc2": "fc2"}
                        if environment[0][1] == "simclr"
                        else {},
                    ),
                    trainer=trainer_config_cls(
                        comment=f"MNIST-Transfer {environment[0][0]}"
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
                "FROMP",
                "FD",
                "FD-MC-Dropout",
                "FD-MC-Dropout-Cov",
            ):
                experiments.append(
                    Experiment(
                        dataset=BaselineDataset(
                            bias=environment[0][0],
                            shuffle=False,
                            valid_size=0.0,
                            convert_to_rgb=("color" in environment[1][0]),
                            filter_classes=split,
                            reduce_to_filtered_classes=False,
                        ),
                        model=BaselineModel(
                            bias=environment[0][0],
                            input_channels=3 if "color" in environment[1][0] else 1,
                            type="lenet5",
                            core_type=environment[0][2],
                            get_intermediate_rep={"fc2": "fc2"}
                            if environment[0][1] == "simclr"
                            else {},
                        ),
                        trainer=transfer_config_cls(
                            comment=f"MNIST Data Generation ({transfer}) {environment[0][0]}",
                        ),
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
                target_dataset = GeneratedDataset
            else:
                target_dataset = BaselineDataset

            # Step 2: Training on bias[1]

            if "split" in environment[1][1]:
                split = tuple(map(int, environment[1][1].split()[1].split("-")))
            else:
                split = ()

            experiments.append(
                Experiment(
                    dataset=target_dataset(
                        dataset_cls="MNIST-Transfer",
                        bias=environment[1][0],
                        filter_classes=split,
                        reduce_to_filtered_classes=False,
                    ),
                    model=BaselineModel(
                        bias=environment[1][0],
                        type="lenet5",
                        core_type=environment[1][2],
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
                        type="lenet5",
                        core_type=environment[2][2],
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
                    name=f"{transfer} ::: {settings} ::: ({environment[0][0]}-{environment[0][2]}->{environment[1][0]}-{environment[1][2]};{environment[2][0]}-{environment[2][2]})",
                    seed=seed,
                )
            ] = TransferExperiment(experiments, update=transfer_settings[transfer])
