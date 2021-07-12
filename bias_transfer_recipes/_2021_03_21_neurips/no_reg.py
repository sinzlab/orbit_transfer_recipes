import math
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


seed = 42
transfer = "NoRegularization"
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
    # (
    #     ("translation_positive", "classification", "conv"),
    #     ("clean", "classification", "lc"),
    #     ("translation_negative", "classification", "lc"),
    # ),
    # (
    #     ("translation_positive", "classification", "conv"),
    #     ("clean", "classification", "fc"),
    #     ("translation_negative", "classification", "fc"),
    # ),
    # (
    #     ("translation_positive", "classification", "conv"),
    #     ("clean", "classification", "fc"),
    #     ("translation", "classification", "fc"),
    # ),
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
    (
        ("clean", "classification", "conv"),
        ("clean", "classification", "fc"),
        ("translation", "classification", "fc"),
    ),
    # (
    #     ("scale", "split-classification 0-4", "conv"),
    #     ("clean", "split-classification 5-9", "conv"),
    #     ("scale", "classification", "conv"),
    # ),
):
    for (dataset_sub_cls, lr,) in product(
        (
            "MNIST",
            "FashionMNIST"
        ),
        (
            0.0003,
            # 0.001, 0.01, 0.00001
        ),  # lr
    ):
        log_prob_loss = False
        experiments = []
        transfer_settings = {
            "NoRegularization": [
                {},
                {},
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
                    dataset_sub_cls=dataset_sub_cls,
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
                    comment=f"MNIST-Transfer {environment[0][0]}", reset="all"
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
                        dataset_sub_cls=dataset_sub_cls,
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
                    dataset_sub_cls=dataset_sub_cls,
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
                    dataset_sub_cls=dataset_sub_cls,
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
                name=f"{transfer} :::{dataset_sub_cls}  {lr} \
                ::: ({environment[0][0]}-{environment[0][2]}->{environment[1][0]}-{environment[1][2]};{environment[2][0]}-{environment[2][2]})",
                seed=seed,
            )
        ] = TransferExperiment(experiments, update=transfer_settings[transfer])
