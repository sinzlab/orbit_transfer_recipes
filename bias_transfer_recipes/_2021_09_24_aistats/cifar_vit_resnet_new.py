from itertools import product

from bias_transfer.configs.model.mnist import TransferModel
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
        self.apply_augmentation: bool = False
        self.apply_normalization: bool = False
        self.input_width: int = 32
        self.input_height: int = 32
        self.batch_size = 128
        super().__init__(**kwargs)


class StudentModel(ClassificationModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "vit"
        self.image_size = 32
        self.patch_size = 4
        self.num_classes = 10
        self.channels = 3
        self.dim = 512
        self.depth = 6
        self.heads = 8
        self.mlp_dim = 512
        self.dropout = 0.1
        self.emb_dropout = 0.1
        self.orig_vit = True
        self.comment = f"CIFAR {self.type}"
        super().__init__(**kwargs)


class TeacherModel(CIFAR10Model):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "resnet18"
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


class BaselineTrainer_(Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
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


class TransferTrainer(TransferMixin, BaselineTrainer_):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        super().__init__(**kwargs)


seed = 42


# rotation_teacher_exp = Experiment(
#     dataset=BaselineDataset(),
#     model=TeacherModel(
#         type="gcnn",
#         get_intermediate_rep={
#             "conv1": "out.1",
#             "conv2": "out.2",
#             "conv3": "out.3",
#             "flatten": "out.4",
#         },
#     ),
#     trainer=BaselineTrainer(comment="Rotation Invariant"),
#     seed=seed,
# )
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

########## Orbit #############
n = 3
id_factor = 10.0
id_between_filters = True
for augmentation, norm, clamp, cut in product(
    [
        True,
        False
    ],
    [True, #False
     ],
    [True, False],
    [True, False],
):
    teacher = Experiment(
        dataset=BaselineDataset(
            apply_augmentation=augmentation, apply_normalization=norm
        ),
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
        Description(
            name=f"MNIST Experiment Teacher {teacher.trainer.comment} (Augment:{augmentation}, Normalization:{norm})",
            seed=seed,
        )
    ] = TransferExperiment([teacher])
    experiments = [teacher]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(
                apply_augmentation=augmentation, apply_normalization=norm
            ),
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
                    "cut_input_grad": clamp,
                    "clamp_input": cut,
                },
                comment="Transfer without fixed identity regularization",
            ),
            seed=seed,
        )
    )

    experiments.append(
        Experiment(
            dataset=BaselineDataset(
                apply_augmentation=augmentation, apply_normalization=norm
            ),
            model=TransferModel(
                spatial_transformer=False,
                num_layers=6,
                vit_input=True,
                first_layer_transform=True,
            ),
            trainer=TransferTrainer(
                student_model=StudentModel(
                    get_intermediate_rep={
                        "transformer.layers.0.1.fn.net.3": "out.1",
                        "transformer.layers.1.1.fn.net.3": "out.2",
                        "transformer.layers.2.1.fn.net.3": "out.3",
                        "transformer.layers.3.1.fn.net.3": "out.4",
                        "transformer.layers.5.1.fn.net.3": "out.5",
                        "mlp_head.1": "out.6",
                    }
                ).to_dict(),
                regularization={
                    "regularizer": "EquivarianceTransfer",
                    "gamma": 1.0,
                    "decay_gamma": False,
                    "group_size": 25,
                    "learn_equiv": False,
                    "max_stacked_transform": n,
                    "cut_input_grad": clamp,
                    "clamp_input": cut,
                },
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{teacher.trainer.comment} Equivariance Transfer (Augment: {augmentation}, Normalization: {norm}, "
            f"repititions:{n}, id_between_filters:{id_between_filters}, id_factor:{id_factor}, clamp:{clamp}, cut:{cut})",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

    experiments = []
    experiments.append(
        Experiment(
            dataset=BaselineDataset(
                apply_augmentation=augmentation, apply_normalization=norm
            ),
            model=StudentModel(),
            trainer=BaselineTrainer(),
            seed=seed,
        )
    )
    transfer_experiments[
        Description(
            name=f"CIFAR Experiment Student (Augment:{augmentation}, Normalization:{norm})",
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