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
        self.batch_norm = True
        super().__init__(**kwargs)


class StudentModel(MLPModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.batch_norm = True
        super().__init__(**kwargs)


class BaselineTrainer(NoiseAugmentationMixin, Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 400
        # self.lr_milestones = (20, 40)
        self.lr_warmup = 20
        self.patience = 20
        self.threshold: float = 0.0
        self.lr_decay_steps = 5  # Number of times the learning rate should be reduced before stopping the training.
        self.lr_decay: float = 0.8
        self.scheduler = "adaptive"
        self.optimizer_options = {
            "amsgrad": False,
            "lr": 0.0001,
            "weight_decay": 0.000000002,
        }
        super().__init__(**kwargs)


class TransferTrainer(TransferMixin, BaselineTrainer):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        super().__init__(**kwargs)


seed = 42
teacher_exp = Experiment(
    dataset=BaselineDataset(),
    model=TeacherModel(),
    trainer=BaselineTrainer(comment="Translation Invariant"),
    seed=seed,
)

rotation_teacher_exp = Experiment(
    dataset=BaselineDataset(),
    model=TeacherModel(type="gcnn"),
    trainer=BaselineTrainer(comment="Rotation Invariant"),
    seed=seed,
)

noise_teacher_exp = Experiment(
    dataset=BaselineDataset(),
    model=TeacherModel(),
    trainer=BaselineTrainer(
        noise_std={
            0.08: 0.1,
            0.12: 0.1,
            0.18: 0.1,
            0.26: 0.1,
            0.38: 0.1,
            -1: 0.5,
        },
        comment="Noise Augmented",
    ),
    seed=seed,
)

for teacher in [teacher_exp, rotation_teacher_exp, noise_teacher_exp]:
    transfer_experiments[
        Description(
            name=f"MNIST Experiment Teacher {teacher.trainer.comment}", seed=seed
        )
    ] = TransferExperiment([teacher])

    ########## Orbit #############
    for n in [1, 2, 3, 4, 5]:
        experiments = [teacher]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(),
                model=TeacherModel(
                    get_intermediate_rep={
                        "layers.0": "out.1",
                        "layers.6": "out.2",
                        "layers.12": "out.3",
                        "layers.20": "out.4",
                    }
                ),
                trainer=TransferTrainer(
                    freeze_bn="all",
                    switch_teacher=True,
                    main_objective="loss",
                    maximize=False,
                    deactivate_dropout=True,
                    student_model=TransferModel().to_dict(),
                    regularization={
                        "regularizer": "EquivarianceTransfer",
                        "gamma": 1.0,
                        "decay_gamma": False,
                        "group_size": 5,
                        "learn_equiv": True,
                        "max_stacked_transform": n,
                    },
                ),
                seed=seed,
            )
        )

        experiments.append(
            Experiment(
                dataset=BaselineDataset(),
                model=TransferModel(),
                trainer=TransferTrainer(
                    student_model=StudentModel(
                        get_intermediate_rep={
                            "layers.1": "out.1",
                            "layers.5": "out.2",
                            "layers.9": "out.3",
                            "layers.13": "out.4",
                        }
                    ).to_dict(),
                    regularization={
                        "regularizer": "EquivarianceTransfer",
                        "gamma": 1.0,
                        "decay_gamma": False,
                        "group_size": 5,
                        "learn_equiv": False,
                        "max_stacked_transform": n,
                    },
                ),
                seed=seed,
            )
        )

        transfer_experiments[
            Description(name=f"{teacher.trainer.comment} Equivariance Transfer ({n} repititions)", seed=seed)
        ] = TransferExperiment(experiments)

    ##### KD ##########
    experiments = [teacher]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=TeacherModel(get_intermediate_rep={"layers.20": "final_out"}),
            trainer=TransferTrainer(
                student_model=StudentModel(
                    get_intermediate_rep={"layers.13": "final_out"}
                ).to_dict(),
                regularization={
                    "regularizer": "KnowledgeDistillation",
                    "gamma": 1.0,
                    "decay_gamma": False,
                    "softmax_temp": 5.0,
                },
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(name=f"{teacher.trainer.comment} Knowledge Distillation", seed=seed)
    ] = TransferExperiment(experiments)

    ##### RDL ##########
    for gamma, dist in product([0.1, 0.5, 0.8], ["CKA", "corr"]):
        experiments = [teacher]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(),
                model=TeacherModel(
                    get_intermediate_rep={
                        "layers.0": "out.1",
                        "layers.6": "out.2",
                        "layers.12": "out.3",
                        "layers.20": "out.4",
                    }
                ),
                trainer=TransferTrainer(
                    student_model=StudentModel(
                        get_intermediate_rep={
                            "layers.1": "out.1",
                            "layers.5": "out.2",
                            "layers.9": "out.3",
                            "layers.13": "out.4",
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
            Description(name=f"{teacher.trainer.comment} RDL ({dist}) gamma:{gamma}", seed=seed)
        ] = TransferExperiment(experiments)

    ##### Attention ##########
    for gamma in [0.1, 0.5, 0.8]:
        experiments = [teacher]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(),
                model=TeacherModel(
                    get_intermediate_rep={
                        "layers.0": "out.1",
                        "layers.6": "out.2",
                        "layers.12": "out.3",
                    }
                ),
                trainer=TransferTrainer(
                    student_model=StudentModel(
                        get_intermediate_rep={
                            "layers.1": "out.1",
                            "layers.5": "out.2",
                            "layers.9": "out.3",
                        }
                    ).to_dict(),
                    regularization={
                        "regularizer": "AttentionTransfer",
                        "gamma": gamma,
                        "decay_alpha": False,
                    },
                ),
                seed=seed,
            )
        )

        transfer_experiments[
            Description(name=f"{teacher.trainer.comment} Attention Transfer gamma:{gamma}", seed=seed)
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
