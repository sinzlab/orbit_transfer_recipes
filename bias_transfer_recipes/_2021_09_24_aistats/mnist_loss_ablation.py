from itertools import product

from bias_transfer.configs.model.mnist import TransferModel
from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from bias_transfer.configs import *

transfer_experiments = {}


class BaselineDataset(MNIST):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.add_corrupted_test = True
        self.add_rotated_test = True
        self.batch_size = 64
        super().__init__(**kwargs)


class TeacherModel(CNNModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.output_size: int = 10
        self.num_layers: int = 4
        self.channels: list = [1, 128, 64, 64]
        self.kernel_size: list = [3, 3, 3]
        self.pool_size: list = [1, 2, 3]
        self.max_out: list = [1, 1, 1]
        self.activation: str = "relu"
        self.dropout: float = 0.1
        self.batch_norm: bool = False
        super().__init__(**kwargs)


class StudentModel(MLPModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.output_size: int = 10
        self.num_layers: int = 4
        self.layer_size: list = [28 * 28, 512, 128, 32]
        self.activation: str = "relu"
        self.dropout: float = 0.1
        self.batch_norm: bool = False
        super().__init__(**kwargs)


class BaselineTrainer(NoiseAugmentationMixin, Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 400
        self.lr_warmup = 20
        self.patience = 20
        self.threshold: float = 0.0
        self.lr_decay_steps = 5  # Number of times the learning rate should be reduced before stopping the training.
        self.lr_decay: float = 0.8
        self.scheduler = "adaptive"
        self.optimizer_options = {
            "amsgrad": False,
            "lr": 0.0003,
            "weight_decay": 0.000000002,
        }
        super().__init__(**kwargs)


class TransferTrainer(TransferMixin, BaselineTrainer):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        super().__init__(**kwargs)


seed = 42
teacher = Experiment(
    dataset=BaselineDataset(),
    model=TeacherModel(
        get_intermediate_rep={
            "layers.0": "out.1",
            "layers.5": "out.2",
            "layers.10": "out.3",
            "layers.17": "out.4",
        }
    ),
    trainer=BaselineTrainer(comment="Translation Invariant"),
    seed=seed,
)

rotation = False
transfer_experiments[
    Description(name=f"MNIST Experiment Teacher {teacher.trainer.comment}", seed=seed)
] = TransferExperiment([teacher])

########## Orbit #############
n = 3
id_between_filters = True
for (first_layer_transform, id_factor, inv_factor, equiv_factor, ce_factor) in product(
    [False, True],
    [10.0, 1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
):
    experiments = [teacher]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=teacher.model,
            trainer=TransferTrainer(
                freeze_bn="all",
                switch_teacher=True,
                main_objective="loss",
                maximize=False,
                deactivate_dropout=True,
                student_model=TransferModel(
                    first_layer_transform=first_layer_transform,
                ).to_dict(),
                regularization={
                    "regularizer": "EquivarianceTransfer",
                    "gamma": 1.0,
                    "decay_gamma": False,
                    "group_size": 25,
                    "learn_equiv": True,
                    "max_stacked_transform": n,
                    "id_between_filters": id_between_filters,
                    "id_factor": id_factor,
                    "inv_factor": inv_factor,
                    "ce_factor": ce_factor,
                    "equiv_factor": equiv_factor,
                    "inv_for_all_layers": False,
                },
                comment="Transfer without fixed identity regularization -> new commit",
            ),
            seed=seed,
        )
    )

    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=TransferModel(
                first_layer_transform=first_layer_transform,
            ),
            trainer=TransferTrainer(
                student_model=StudentModel(
                    get_intermediate_rep={
                        "layers.1": "out.1",
                        "layers.4": "out.2",
                        "layers.7": "out.3",
                        "layers.10": "out.4",
                    }
                ).to_dict(),
                regularization={
                    "regularizer": "EquivarianceTransfer",
                    "gamma": 1.0,
                    "decay_gamma": False,
                    "group_size": 25,
                    "learn_equiv": False,
                    "max_stacked_transform": n,
                },
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{teacher.trainer.comment} Equivariance Transfer "
            f"(first_layer_transform: {first_layer_transform}, id:{id_factor}, equiv:{equiv_factor}, inv:{inv_factor}, ce:{ce_factor})",
            seed=seed,
        )
    ] = TransferExperiment(experiments)
