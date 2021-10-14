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
        self.hidden_dim: int = 600
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


seed = 42
teacher_exp = Experiment(
    dataset=BaselineDataset(),
    model=TeacherModel(),
    trainer=BaselineTrainer(),
    seed=seed,
)
for forward, lr, hidden_dim, weight_decay in product(
    ("ce",),
    (0.001, 0.0001),
    (40, 80, 120, 200, 600, 400, 800),
    (1e-4, 1e-6, 1e-8),
):
    experiments = [teacher_exp]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=TeacherModel(),
            trainer=BaselineTrainer(
                student_model=StudentModel(hidden_dim=hidden_dim).to_dict(),
                forward=forward,
                learning_rate=lr,
                weight_decay=weight_decay,
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{forward}: lr={lr} weight_decay={weight_decay} hidden_dim={hidden_dim}",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

for forward, lr, hidden_dim, weight_decay, gamma, softmax_temp in product(
    ("kd",),
    (0.001, 0.0001),
    (600,),
    (1e-4, 1e-6, 1e-8),
    list(np.linspace(0.1, 1.0, 10)),
    # [1.0],
    (1.0, 2.0, 5.0, 10.0),
):
    experiments = [teacher_exp]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=TeacherModel(),
            trainer=BaselineTrainer(
                student_model=StudentModel(hidden_dim=hidden_dim).to_dict(),
                forward=forward,
                learning_rate=lr,
                gamma=gamma,
                softmax_temp=softmax_temp,
                weight_decay=weight_decay,
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{forward}: gamma={gamma} T={softmax_temp} lr={lr} weight_decay={weight_decay} hidden_dim={hidden_dim}",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

for forward, lr, weight_decay, gamma, softmax_temp in product(
    ("kd_match",),
    (0.001, 0.0001),
    (1e-4, 1e-6, 1e-8),
    list(np.linspace(0.1, 1.0, 10)),
    # [1.0],
    (1.0, 2.0, 5.0, 10.0),
):
    experiments = [teacher_exp]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=TeacherModel(),
            trainer=BaselineTrainer(
                student_model=StudentModel().to_dict(),
                forward=forward,
                learning_rate=lr,
                gamma=gamma,
                softmax_temp=softmax_temp,
                weight_decay=weight_decay,
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{forward}: gamma={gamma} T={softmax_temp} lr={lr} weight_decay={weight_decay}",
            seed=seed,
        )
    ] = TransferExperiment(experiments)
"""
   Train  Validation  Validation Shift  Test Shift  gamma     T    lr
name                                                                        
kd    98.611111       52.75              15.5        14.6    0.7  10.0  0.01
kd_match  100.0        98.0             96.75        96.8    0.1  0.1  0.0005
"""

for forward, lr, gamma in product(
    (
        "rdl",
        "cka",
        "attention",
    ),
    (0.01, 0.001, 0.0001, 0.0005),
    list(np.linspace(0.1, 1.0, 10)),
):
    experiments = [teacher_exp]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=TeacherModel(),
            trainer=BaselineTrainer(
                student_model=StudentModel().to_dict(),
                forward=forward,
                learning_rate=lr,
                gamma=gamma,
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{forward}: gamma={gamma} lr={lr}",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

"""
               Train  Validation  Validation Shift  Test Shift  gamma    lr
name                                                                       
attention  98.805556       51.25              15.5        13.8    0.7  0.01
cka             100.0        59.5             18.25        15.8    0.7  0.0001
rdl              99.75        56.5             10.25        13.2    0.8  0.0001
"""

for lr, gamma, equiv, inv, id in product(
    (0.01, 0.001, 0.0001, 0.0005),
    list(np.linspace(0.1, 1.0, 10)),
    (0.1, 1.0, 10.0),
    (0.1, 1.0, 10.0),
    (0.1, 1.0, 10.0),
):
    experiments = [teacher_exp]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=TeacherModel(),
            trainer=BaselineTrainer(
                student_model=EquivTransferModel().to_dict(),
                forward="equiv_learn",
                learning_rate=lr,
                equiv_factor=equiv,
                invertible_factor=inv,
                identity_factor=id,
                select_on_loss=True,
                id_between_filters=True,
            ),
            seed=seed,
        )
    )
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=EquivTransferModel(),
            trainer=BaselineTrainer(
                student_model=StudentModel().to_dict(),
                forward="equiv_transfer",
                learning_rate=lr,
                gamma=gamma,
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"Equiv transfer: gamma={gamma} lr={lr} equiv={equiv} inv={inv} id={id}",
            seed=seed,
        )
    ] = TransferExperiment(experiments)


"""
 Train  Validation  Validation + Shift  Test + Shift  \
name                                                                      
Equiv transfer  96.638889       97.25               97.25          97.2   

                gamma    lr  equiv  inv   id  
name                                          
Equiv transfer    1.0  0.01    1.0  1.0  1.0  
"""
