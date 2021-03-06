from itertools import product

from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from orbit_transfer.configs import *

transfer_experiments = {}


class BaselineDataset(MNIST):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.add_corrupted_test = True
        self.add_rotated_test = True
        self.batch_size = 256
        super().__init__(**kwargs)


class StudentModel(ClassificationModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "vit"
        self.image_size = 28
        self.patch_size = 7
        self.num_classes = 10
        self.channels = 1
        self.dim = 64
        self.depth = 6
        self.heads = 8
        self.mlp_dim = 128
        self.comment = f"MNIST {self.type}"
        super().__init__(**kwargs)


class TeacherModel(MNISTModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "resnet18"
        super().__init__(**kwargs)


class BaselineTrainer(Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 1
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
            "conv1": "out.1",
            "layer1.1.conv2": "out.2",
            "layer2.1.conv2": "out.3",
            "layer3.1.conv2": "out.4",
            "layer4.1.conv2": "out.5",
            "fc": "out.6",
        }
    ),
    trainer=BaselineTrainer(comment="Translation Invariant"),
    seed=seed,
)

rotation = teacher.model.type == "gcnn"
transfer_experiments[
    Description(name=f"MNIST Experiment Teacher {teacher.trainer.comment}", seed=seed)
] = TransferExperiment([teacher])

########## Orbit #############
for n, id_between_filters, id_factor in product(
    [
        3,  # 2, 3, 4, 5
    ],
    [
        True,
        # False
    ],
    [
        # 0.1,
        # 1.0,
        10.0,
        # 100.0
    ],
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
                    spatial_transformer=False,
                    num_layers=6,
                    vit_input=False,
                    first_layer_transform=True,
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
                    "cut_input_grad": True,
                    "clamp_input": True,
                },
                comment="Transfer without fixed identity regularization",
            ),
            seed=seed,
        )
    )

    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=TransferModel(
                spatial_transformer=False,
                num_layers=6,
                vit_input=True,
                first_layer_transform=True,
            ),
            trainer=TransferTrainer(
                student_model=StudentModel(
                    get_intermediate_rep={
                        "transformer.layers.0.1.fn.net.4": "out.1",
                        "transformer.layers.1.1.fn.net.4": "out.2",
                        "transformer.layers.2.1.fn.net.4": "out.3",
                        "transformer.layers.3.1.fn.net.4": "out.4",
                        "transformer.layers.5.1.fn.net.4": "out.5",
                        "mlp_head.2": "out.6",
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
            name=f"{teacher.trainer.comment} Equivariance Transfer (repititions:{n}, "
            f"id_between_filters:{id_between_filters}, id_factor:{id_factor})",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

##### KD ##########
for softmax_temp in [5.0]:
    experiments = [teacher]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=teacher.model,
            trainer=TransferTrainer(
                student_model=StudentModel(
                    get_intermediate_rep={
                        "mlp_head.2": "out.6",
                    }
                ).to_dict(),
                regularization={
                    "regularizer": "KnowledgeDistillation",
                    "gamma": 1.0,
                    "decay_gamma": False,
                    "softmax_temp": softmax_temp,
                },
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{teacher.trainer.comment} Knowledge Distillation T={softmax_temp}",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

##### RDL ##########
for gamma, dist in product(
    [0.8],
    ["corr"],
):
    experiments = [teacher]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=teacher.model,
            trainer=TransferTrainer(
                student_model=StudentModel(
                    get_intermediate_rep={
                        "transformer.layers.0.1.fn.net.4": "out.1",
                        "transformer.layers.1.1.fn.net.4": "out.2",
                        "transformer.layers.2.1.fn.net.4": "out.3",
                        "transformer.layers.3.1.fn.net.4": "out.4",
                        "transformer.layers.5.1.fn.net.4": "out.5",
                        "mlp_head.2": "out.6",
                    }
                ).to_dict(),
                regularization={
                    "regularizer": "RDL",
                    "gamma": gamma,
                    "decay_gamma": False,
                    "dist_measure": dist,
                },
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{teacher.trainer.comment} RDL ({dist}) gamma:{gamma}", seed=seed
        )
    ] = TransferExperiment(experiments)


experiments = []
experiments.append(
    Experiment(
        dataset=BaselineDataset(),
        model=StudentModel(),
        trainer=BaselineTrainer(),
        seed=seed,
    )
)
transfer_experiments[
    Description(name=f"MNIST Experiment Student", seed=seed)
] = TransferExperiment(experiments)

experiments = []
experiments.append(
    Experiment(
        dataset=BaselineDataset(dataset_cls="MNIST-C"),
        model=StudentModel(),
        trainer=BaselineTrainer(comment="Training on MNIST-C"),
        seed=seed,
    )
)
transfer_experiments[
    Description(name=f"MNIST-C Experiment Student", seed=seed)
] = TransferExperiment(experiments)
