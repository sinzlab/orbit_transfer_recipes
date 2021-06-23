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
transfer = "FD-MC-Dropout-Cov"
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
        ("translation_negative", "classification", "lc"),
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
    (
            ("translation", "classification", "conv"),
            ("clean", "classification", "fc"),
            ("translation", "classification", "fc"),
    ),
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
    for (
        intial_std,
        (dropout, ensemble_members),
        regularize_mean,
        (penultimate, marginalize_over_hidden, use_softmax),
        lr,
    ) in product(
        (
                1.0, 0.1,
         0.01,
         0.001, 0.0001,
            0.000001
         ),  # intial std
        (
            (0.0, 5),
            (0.1, 10),
            (0.1, 40),
            (0.3, 10),
            (0.5, 10),
        ),  # dropout, ensemble_members
        (True,
         False
         ),  # reularize_mean
        (
            (True, True, False),
            (
                False,
                True,
                True,
            ),
            (
                False,
                True,
                False,
            ),
            (
                True,
                False,
                False,
            ),
        ),  # (penultimate,marginalize_over_hidden,softmax)
        (0.0003,
         # 0.001, 0.01, 0.00001
         ),  # lr
    ):
        readout_layer = "fc2" if penultimate else "fc3"
        ensembling = dropout == 0.0
        log_prob_loss = True
        log_var = math.log(intial_std ** 2)
        experiments = []
        transfer_settings = {
            "FD-MC-Dropout-Cov": [
                {
                    "model": {"dropout": dropout},
                },
            ]
            * (ensemble_members if ensembling else 1)
            + [
                {
                    "model": {
                        "get_intermediate_rep": {readout_layer: readout_layer},
                        "dropout": dropout,
                    },
                    "trainer": {
                        "save_representation": True,
                        "save_input": True,
                        "data_transfer": True,
                        "apply_softmax": use_softmax,
                        "softmax_temp": 1.0,
                        "compute_covariance": {
                            "type": "full",
                            "precision": "double",
                            "n_components": ensemble_members,
                            "n_samples": ensemble_members,
                            "ensembling": ensembling,
                        },
                    },
                },
                {
                    "model": {
                        "get_intermediate_rep": {readout_layer: readout_layer},
                        # "add_custom_buffer": {"fc3_cov_lambdas": (10,)},
                    },
                    "trainer": {
                        "reset": "all",
                        "single_input_stream": False,
                        "regularization": {
                            "regularizer": "FunctionDistance",
                            "alpha": 1.0,
                            "decay_alpha": False,
                            "softmax_temp": 1.0,
                            "use_softmax": use_softmax,
                            "log_var": log_var,
                            "learn_log_var": True,
                            "marginalize_over_hidden": marginalize_over_hidden,
                            "regularize_mean": regularize_mean,
                            "add_determinant": log_prob_loss,
                        },
                        "loss_functions": {
                            "img_classification": "CELikelihood"
                            if log_prob_loss
                            else "CrossEntropyLoss"
                        },
                        "loss_function_options": {
                            "img_classification": {"log_var": log_var}
                            if log_prob_loss
                            else {}
                        },
                        "data_transfer": True,
                        "ignore_main_loss": False,
                        "optim_step_count": 2,
                        "optimizer_options": {
                            "amsgrad": False,
                            "lr": lr,
                        },
                    },
                },
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
        if ensembling:
            for i in range(0, ensemble_members - 1):
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
                            add_buffer=tuple([f"ensemble_{j}" for j in range(i)]),
                        ),
                        trainer=trainer_config_cls(
                            comment=f"MNIST-Transfer {environment[0][0]}",
                            ensemble_iteration=i,
                            reset="all",
                        ),
                        seed=seed + i + 1,
                    )
                )

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
                    add_buffer=tuple(
                        [f"ensemble_{i}" for i in range(ensemble_members - 1)]
                    )
                    if ensembling
                    else (),
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
                    ),
                    model=BaselineModel(
                        bias=environment[0][0],
                        input_channels=3 if "color" in environment[1][0] else 1,
                        type="lenet5",
                        core_type=environment[0][2],
                        get_intermediate_rep={"fc2": "fc2"}
                        if environment[0][1] == "simclr"
                        else {},
                        add_buffer=tuple(
                            [f"ensemble_{i}" for i in range(ensemble_members - 1)]
                        )
                        if ensembling
                        else (),
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
                name=f"{transfer} ::: {intial_std},{dropout},{ensemble_members},{regularize_mean},{penultimate}, {marginalize_over_hidden},{use_softmax},{lr} \
                ::: ({environment[0][0]}-{environment[0][2]}->{environment[1][0]}-{environment[1][2]};{environment[2][0]}-{environment[2][2]})",
                seed=seed,
            )
        ] = TransferExperiment(experiments, update=transfer_settings[transfer])
