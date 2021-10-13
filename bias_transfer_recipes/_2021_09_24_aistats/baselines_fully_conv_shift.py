from itertools import product
import numpy as np

from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from bias_transfer.configs import *

transfer_experiments = {}


class BaselineDataset(MNIST1DDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.train_shift = 30
        super().__init__(**kwargs)


class TeacherModel(MNIST1DModelConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type = "fully_conv_single"
        self.kernel_size = 5
        self.padding = 2
        self.input_size = 40
        self.layers = 2
        self.channels = 15
        super().__init__(**kwargs)


class StudentModel(TeacherModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type = "fc_single"
        super().__init__(**kwargs)


class EquivTransferModel(MNIST1DModelConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type = "static_equivariance_transfer"
        self.kernel_size = 40
        self.layers = 3
        self.group_size = 40
        super().__init__(**kwargs)


class BaselineTrainer(SimpleTrainerConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.total_steps = 40000
        self.eval_every = 100
        self.print_every = 100
        self.learning_rate = 0.01
        self.patience = 20
        self.lr_decay_steps = 5
        super().__init__(**kwargs)


for shift in [
    30,
]:  # np.linspace(0, 40, 9)
    seed = 42
    teacher_exp = Experiment(
        dataset=BaselineDataset(train_shift=shift),
        model=TeacherModel(),
        trainer=BaselineTrainer(comment="Teacher"),
        seed=seed,
    )

    settings = {
        "kd": {"lr": 0.001, "softmax_temp": 1, "gamma": 1.0},
        # "kd_match": {"lr": 0.0005, "softmax_temp": 0.1, "gamma": 0.1},
        # "attention": {"lr": 0.001, "gamma": 0.5},
        # "cka": {"lr": 0.001, "gamma": 0.5},
        # "rdl": {"lr": 0.001, "gamma": 0.5},
        "ce": {"lr": 0.001}
    }

    for forward, setting in settings.items():
        experiments = [teacher_exp]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(train_shift=shift),
                model=TeacherModel(),
                trainer=BaselineTrainer(
                    student_model=StudentModel(hidden_dim=80).to_dict(),
                    forward=forward,
                    weight_decay=1e-6,
                    **setting,
                ),
                seed=seed,
            )
        )

        transfer_experiments[
            Description(
                name=f"{forward}: shift={shift} "
                + " ".join([f"{k}={v}" for k, v in setting.items()]),
                seed=seed,
            )
        ] = TransferExperiment(experiments)

    # lr = 0.01
    # gamma = 1.0
    # inv = 1.0
    # equiv = 1.0
    # id = 1.0
    #
    # experiments = [teacher_exp]
    # experiments.append(
    #     Experiment(
    #         dataset=BaselineDataset(train_shift=shift),
    #         model=TeacherModel(),
    #         trainer=BaselineTrainer(
    #             student_model=EquivTransferModel().to_dict(),
    #             forward="equiv_learn",
    #             learning_rate=lr,
    #             equiv_factor=equiv,
    #             invertible_factor=inv,
    #             identity_factor=id,
    #             select_on_loss=True
    #         ),
    #         seed=seed,
    #     )
    # )
    # experiments.append(
    #     Experiment(
    #         dataset=BaselineDataset(train_shift=shift),
    #         model=EquivTransferModel(),
    #         trainer=BaselineTrainer(
    #             student_model=StudentModel().to_dict(),
    #             forward="equiv_transfer",
    #             learning_rate=lr,
    #             gamma=gamma,
    #         ),
    #         seed=seed,
    #     )
    # )
    #
    # transfer_experiments[
    #     Description(
    #         name=f"Equiv transfer: shift={shift} gamma={gamma} lr={lr} equiv={equiv} inv={inv} id={id}",
    #         seed=seed,
    #     )
    # ] = TransferExperiment(experiments)
