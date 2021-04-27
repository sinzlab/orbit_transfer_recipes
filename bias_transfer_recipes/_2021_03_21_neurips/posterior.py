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


seed = 42
for environment in (
    (
        ("clean", "classification", "mlp"),
        ("color", "classification", "mlp"),
        ("color_shuffle", "classification", "mlp"),
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
        ("scale", "split-classification 0-4", "mlp"),
        ("clean", "split-classification 5-9", "mlp"),
        ("scale", "classification", "mlp"),
    ),
):
    for transfer, alphas, resets in (
        (
            "ELRG L2-SP",
            (1.0,),
            ("",),
        ),
        (
            "MF L2-SP",
            (1.0,),
            ("",),
        ),
        (
            "VCL",
            (1.0,),
            ("",),
        ),
    ):
        for alpha in alphas:
            for reset in resets:
                experiments = []
                softmax_temp = 1.0
                rank = 8
                transfer_settings = {
                    "VCL": [
                        {},
                        {
                            "trainer": {
                                "regularization": {
                                    "regularizer": "VCL",
                                    "alpha": alpha,
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
                                "reset": reset,
                                "regularization": {
                                    "regularizer": "VCL",
                                    "alpha": alpha,
                                },
                            },
                        },
                    ],
                    "MF L2-SP": [
                        {},
                        {
                            "trainer": {
                                "regularization": {
                                    "regularizer": "VCL",
                                    "alpha": alpha,
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
                                "reset": reset,
                                "regularization": {
                                    "regularizer": "ParamDistance",
                                    "alpha": alpha,
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
                        {},
                        {
                            "model": {
                                "type": "lenet300-100-elrg",
                                "alpha": None,
                                "rank": rank,
                            },
                            "trainer": {
                                "regularization": {
                                    "regularizer": "ELRG",
                                    "alpha": alpha,
                                },
                            },
                        },
                        {
                            "model": {
                                "type": "lenet300-100-elrg",
                                "alpha": None,
                                "rank": rank,
                            },
                            "trainer": {
                                "bayesian_to_deterministic": True,
                                "reset_for_new_task": False,
                            },
                        },
                        {
                            "model": {
                                "type": "lenet300-100",
                                "add_buffer": ("importance", ("importance_v", rank)),
                            },
                            "trainer": {
                                "reset": reset,
                                "regularization": {
                                    "regularizer": "ParamDistance",
                                    "alpha": alpha,
                                    "use_elrg_importance": True,
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
                }

                if (
                    transfer == "KnowledgeDistillation"
                    and "regression" in environment[0][1]
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

                # Step 1: Normal training on source_bias
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
                            type="lenet300-100",
                            get_intermediate_rep={"fc2": "fc2"}
                            if environment[0][1] == "simclr"
                            else {},
                        ),
                        trainer=trainer_config_cls(
                            comment=f"MNIST-Transfer Normal training on {environment[0][0]}"
                        ),
                        seed=seed,
                    )
                )

                # Step 3: Training on source_bias with VCL Loss
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
                            type="lenet300-100-bayes",
                            get_intermediate_rep={"fc2": "fc2"}
                            if environment[0][1] == "simclr"
                            else {},
                        ),
                        trainer=trainer_config_cls(
                            comment=f"MNIST-Transfer {environment[0][0]}",
                            data_transfer=True,
                        ),
                        seed=seed,
                    )
                )

                # Step 4: Move parameters to prior
                experiments.append(
                    Experiment(
                        dataset=BaselineDataset(
                            bias=environment[1][0],
                            shuffle=False,
                            valid_size=0.0,
                            filter_classes=split,
                            reduce_to_filtered_classes=False,
                        ),
                        model=BaselineModel(
                            bias=environment[1][0],
                            input_channels=3 if "color" in environment[1][0] else 1,
                            type="lenet300-100-bayes",
                            get_intermediate_rep={"fc2": "fc2"}
                            if environment[1][1] == "simclr"
                            else {},
                        ),
                        trainer=transfer_config_cls(
                            comment=f"MNIST Update Coreset ({transfer}) {environment[1][0]}",
                        ),
                        seed=seed,
                    )
                )

                # Step 5: Training on target bias with VCL loss
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
                            type="lenet300-100-bayes",
                        ),
                        trainer=BaselineTrainer(
                            comment=f"MNIST Transfer ({transfer}) {environment[1][0]}",
                        ),
                        seed=seed,
                    )
                )

                # Step 6: Test on bias[2]
                experiments.append(
                    Experiment(
                        dataset=BaselineDataset(
                            bias=environment[2][0],
                        ),
                        model=BaselineModel(
                            bias=environment[2][0],
                            input_channels=3 if "color" in environment[1][0] else 1,
                            type="lenet300-100-bayes",
                        ),
                        trainer=BaselineTrainer(
                            comment=f"Test MNIST-Transfer {environment[2][0]}",
                            max_iter=0,
                        ),
                        seed=seed,
                    )
                )

                reset_string = "reset" if reset == "all" else ""
                transfer_experiments[
                    Description(
                        name=f"{transfer} {reset_string}: {alpha} ({environment[0][0]}->{environment[1][0]};{environment[2][0]})",
                        seed=seed,
                    )
                ] = TransferExperiment(experiments, update=transfer_settings[transfer])
