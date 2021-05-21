from itertools import product

from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from bias_transfer.configs import *

transfer_experiments = {}


class BaselineDataset(ImageDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "CIFAR10"
        super().__init__(**kwargs)


class BaselineModel(CIFAR10Model):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.coreset_size = 50
        self.type: str = "lenet5"
        super().__init__(**kwargs)


class BaselineTrainer(TransferMixin, Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.patience = 1000
        super().__init__(**kwargs)


class DataGenerator(DataGenerationMixin, Classification):
    fn = "bias_transfer.trainer.transfer"


class GeneratedDataset(Generated, BaselineDataset):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.primary_dataset_fn = "bias_transfer.dataset.mnist_transfer_dataset_loader"
        super().__init__(**kwargs)


possible_settings = {
    "FD-MC-Dropout-Cov": (
        (0.001,),  # alpha
        (
            # (0.0, 5),
            (0.1, 2),
            # (0.3, 40), (0.5, 40)
        ),  # dropout, ensemble_members
        (1e-12,),  # eps
        (
            True,
            # False
        ),  # reularize_mean
        (
            (True, True, False),
            # (
            #     False,
            #     True,
            #     True,
            # ),
            # (
            #     False,
            #     True,
            #     False,
            # ),
            # (
            #     True,
            #     False,
            #     False,
            # ),
        ),  # (penultimate,marginalize_over_hidden,softmax)
        ("conv",),
    ),
}

seed = 9
for transfer in (
    # "Finetune",
    # "FROMP",
    # "FD",
    # "FD-MC-Dropout",
    "FD-MC-Dropout-Cov",
    # "KnowledgeDistillation",
):
    for (
        alpha,
        (dropout, ensemble_members),
        eps,
        regularize_mean,
        (penultimate, marginalize_over_hidden, use_softmax),
        core_type,
    ) in product(*possible_settings[transfer]):
        readout_layer = "fc2" if penultimate else "fc3"
        ensembling = dropout == 0.0
        log_prob_loss = alpha == 1.0
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
                            "eps": eps,
                            "n_components": ensemble_members,
                            "n_samples": ensemble_members,
                            "ensembling": ensembling,
                        },
                    },
                },
                # {
                #     "model": {
                #         "get_intermediate_rep": {readout_layer: readout_layer},
                #         # "add_custom_buffer": {"fc3_cov_lambdas": (10,)},
                #     },
                #     "trainer": {
                #         "reset": "all",
                #         "single_input_stream": False,
                #         "regularization": {
                #             "regularizer": "FunctionDistance",
                #             "alpha": alpha if alpha != -1 else 1.0,
                #             "decay_alpha": False,
                #             "softmax_temp": 1.0,
                #             "use_softmax": use_softmax,
                #             "cov_eps": eps,
                #             "marginalize_over_hidden": marginalize_over_hidden,
                #             "regularize_mean": regularize_mean,
                #             "add_determinant": log_prob_loss,
                #         },
                #         "loss_functions": {
                #             "img_classification": "CELikelihood"
                #             if log_prob_loss
                #             else "CrossEntropyLoss"
                #         },
                #         "data_transfer": True,
                #         "ignore_main_loss": alpha == -1,
                #         "optim_step_count": 2 if alpha != -1 else 1,
                #     },
                # },
            ],
        }

        trainer_config_cls = BaselineTrainer
        transfer_config_cls = DataGenerator

        # Step 1: Training on source_bias
        if ensembling:
            for i in range(0, ensemble_members - 1):
                experiments.append(
                    Experiment(
                        dataset=BaselineDataset(),
                        model=BaselineModel(
                            core_type=core_type,
                            add_buffer=tuple([f"ensemble_{j}" for j in range(i)]),
                        ),
                        trainer=trainer_config_cls(
                            comment=f"CIFAR ensemble member {i}",
                            ensemble_iteration=i,
                            reset="all",
                        ),
                        seed=seed + i + 1,
                    )
                )

        # Step 1: Training on source_bias
        experiments.append(
            Experiment(
                dataset=BaselineDataset(),
                model=BaselineModel(
                    core_type=core_type,
                    add_buffer=tuple(
                        [f"ensemble_{i}" for i in range(ensemble_members - 1)]
                    )
                    if ensembling
                    else (),
                ),
                trainer=trainer_config_cls(comment=f"CIFAR", reset="all"),
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
                    dataset=BaselineDataset(shuffle=False, valid_size=0.0),
                    model=BaselineModel(
                        type="lenet5",
                        core_type=core_type,
                        add_buffer=tuple(
                            [f"ensemble_{i}" for i in range(ensemble_members - 1)]
                        )
                        if ensembling
                        else (),
                    ),
                    trainer=transfer_config_cls(
                        comment=f"MNIST Data Generation ({transfer})"
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
        #
        # # Step 2: Training on bias[1]
        # experiments.append(
        #     Experiment(
        #         dataset=target_dataset(
        #             dataset_cls="MNIST-Transfer",
        #         ),
        #         model=BaselineModel(
        #             type="lenet5",
        #             core_type=core_type,
        #         ),
        #         trainer=BaselineTrainer(
        #             comment=f"CIFAR Transfer ({transfer})",
        #         ),
        #         seed=seed,
        #     )
        # )
        #
        # # Step 3: Test on bias[2]
        # experiments.append(
        #     Experiment(
        #         dataset=BaselineDataset(
        #             bias=environment[2][0],
        #         ),
        #         model=BaselineModel(
        #             bias=environment[2][0],
        #             input_channels=3 if "color" in environment[1][0] else 1,
        #             type="lenet5",
        #             core_type=environment[2][2],
        #         ),
        #         trainer=BaselineTrainer(
        #             comment=f"Test MNIST-Transfer {environment[2][0]}",
        #             max_iter=0,
        #         ),
        #         seed=seed,
        #     )
        # )

        transfer_experiments[
            Description(
                name=f"{transfer} ::: {alpha} {dropout} {ensemble_members} {eps} {regularize_mean} {penultimate} {marginalize_over_hidden} {use_softmax} {core_type}:::",
                seed=seed,
            )
        ] = TransferExperiment(experiments, update=transfer_settings[transfer])
