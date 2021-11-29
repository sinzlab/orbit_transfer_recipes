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
        # self.max_iter = 20
        self.max_iter = 40
        # self.lr_warmup = 20
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
        self.noise_test = {}
        super().__init__(**kwargs)


class TransferTrainer(TransferMixin, BaselineTrainer):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 30
        super().__init__(**kwargs)


seed = 42
teacher_exp = Experiment(
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

rotation_teacher_exp = Experiment(
    dataset=BaselineDataset(),
    model=TeacherModel(
        type="gcnn",
        small=False,
        large_filters=False,
        get_intermediate_rep={
            "pool1": "out.1",
            "pool2": "out.2",
            "pool3": "out.3",
            "flatten": "out.4",
        },
    ),
    trainer=BaselineTrainer(comment="Rotation Invariant"),
    seed=seed,
)
#
# noise_teacher_exp = Experiment(
#     dataset=BaselineDataset(),
#     model=StudentModel(
#         get_intermediate_rep={
#             "layers.1": "out.1",
#             "layers.4": "out.2",
#             "layers.7": "out.3",
#             "layers.10": "out.4",
#         }
#     ),
#     trainer=BaselineTrainer(
#         noise_std={
#             0.08: 0.1,
#             0.12: 0.1,
#             0.18: 0.1,
#             0.26: 0.1,
#             0.38: 0.1,
#             -1: 0.5,
#         },
#         comment="Noise Augmented",
#     ),
#     seed=seed,
# )
#
G = 4
for teacher in [
    # teacher_exp,
    rotation_teacher_exp,
    # noise_teacher_exp,
]:
    rotation = teacher.model.type == "gcnn"
    # transfer_experiments[
    #     Description(
    #         name=f"MNIST Experiment Teacher {teacher.trainer.comment}", seed=seed
    #     )
    # ] = TransferExperiment([teacher])

    ########## Orbit #############
    for n, id_between_filters, id_factor in product(
        [
            1,  # 2, 3, 4, 5
        ],
        [
            # True,
            False
        ],
        [
            # 0.1,
            1.0,
            # 10.0,
            # 100.0
        ],
    ):
        experiments = [teacher]
        seed = 44
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
                        # unet=True,
                        spatial_transformer=True,
                        only_translation=False,
                        prevent_translation=False,
                        include_channels=True,
                        use_layer_transforms=True,
                        # kernel_size=5,
                        # spatial_kernel_size=[5, 5, 3, 1],
                        # channel_kernel_size=[1, 2, 4, 5],
                        # spatial_pooling=[[2, 2], [2, 2], [2, 2], [1] * 2],
                        # channel_pooling=[[1] * 2, [1] * 2, [2, 2], [1, 1]],
                        # latent_size=5,
                        num_layers=4,
                        group_size=G,
                        patch_size=12,
                    ).to_dict(),
                    regularization={
                        "regularizer": "EquivarianceTransfer",
                        "gamma": 1.0,
                        "decay_gamma": False,
                        "group_size": G,
                        "learn_equiv": True,
                        "max_stacked_transform": n,
                        "id_between_filters": True,
                        "id_between_transforms": False,
                        "id_factor": 1.0,
                        "ce_factor": 1.0,
                        "equiv_factor": 1.0,
                        "inv_factor": 1.0,
                        "hinge_epsilon": 1.5,
                        "mse_dist": True,
                        "ramp_up": {"inv": 10, "equiv": 10}
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
                    spatial_transformer=True,
                    only_translation=False,
                    prevent_translation=False,
                    include_channels=True,
                    use_layer_transforms=True,
                    patch_size=32,
                    num_layers=4,
                    group_size=G,
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
                        "group_size": G,
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
#
#     ##### KD ##########
#     for softmax_temp in [0.1, 1.0, 2.0, 5.0, 10.0]:
#         experiments = [teacher]
#         experiments.append(
#             Experiment(
#                 dataset=BaselineDataset(),
#                 model=teacher.model,
#                 trainer=TransferTrainer(
#                     student_model=StudentModel(
#                         get_intermediate_rep={"layers.10": "out.4"}
#                     ).to_dict(),
#                     regularization={
#                         "regularizer": "KnowledgeDistillation",
#                         "gamma": 1.0,
#                         "decay_gamma": False,
#                         "softmax_temp": softmax_temp,
#                     },
#                 ),
#                 seed=seed,
#             )
#         )
#
#         transfer_experiments[
#             Description(
#                 name=f"{teacher.trainer.comment} Knowledge Distillation T={softmax_temp}",
#                 seed=seed,
#             )
#         ] = TransferExperiment(experiments)
#
#     ##### RDL ##########
#     for gamma, dist in product(
#         [0.1, 0.5, 0.8,
#          0.9],
#         ["CKA",
#          "corr"],
#     ):
#         experiments = [teacher]
#         experiments.append(
#             Experiment(
#                 dataset=BaselineDataset(),
#                 model=teacher.model,
#                 trainer=TransferTrainer(
#                     student_model=StudentModel(
#                         get_intermediate_rep={
#                             "layers.1": "out.1",
#                             "layers.4": "out.2",
#                             "layers.7": "out.3",
#                             "layers.10": "out.4",
#                         }
#                     ).to_dict(),
#                     regularization={
#                         "regularizer": "RDL",
#                         "gamma": gamma,
#                         "decay_gamma": False,
#                         "dist_measure": dist,
#                     },
#                 ),
#                 seed=seed,
#             )
#         )
#
#         transfer_experiments[
#             Description(
#                 name=f"{teacher.trainer.comment} RDL ({dist}) gamma:{gamma}", seed=seed
#             )
#         ] = TransferExperiment(experiments)
#
#     #### Attention ##########
#     for gamma in [0.1, 0.5, 0.8, 0.9]:
#         experiments = [teacher]
#         experiments.append(
#             Experiment(
#                 dataset=BaselineDataset(),
#                 model=teacher.model,
#                 trainer=TransferTrainer(
#                     student_model=StudentModel(
#                         get_intermediate_rep={
#                             "layers.1": "out.1",
#                             "layers.4": "out.2",
#                             "layers.7": "out.3",
#                         }
#                     ).to_dict(),
#                     regularization={
#                         "regularizer": "AttentionTransfer",
#                         "gamma": gamma,
#                         "decay_alpha": False,
#                     },
#                 ),
#                 seed=seed,
#             )
#         )
#
#         transfer_experiments[
#             Description(
#                 name=f"{teacher.trainer.comment} Attention Transfer gamma:{gamma}",
#                 seed=seed,
#             )
#         ] = TransferExperiment(experiments)
#
# experiments = []
# experiments.append(
#     Experiment(
#         dataset=BaselineDataset(),
#         model=StudentModel(),
#         trainer=BaselineTrainer(),
#         seed=seed,
#     )
# )
# transfer_experiments[
#     Description(name=f"MNIST Experiment Student", seed=seed)
# ] = TransferExperiment(experiments)
#
# experiments = []
# experiments.append(
#     Experiment(
#         dataset=BaselineDataset(dataset_cls="MNIST-C"),
#         model=StudentModel(),
#         trainer=BaselineTrainer(comment="Training on MNIST-C"),
#         seed=seed,
#     )
# )
# transfer_experiments[
#     Description(name=f"MNIST-C Experiment Student", seed=seed)
# ] = TransferExperiment(experiments)
