from itertools import product
import numpy as np

from nntransfer.configs.base import Description
from nntransfer.configs.transfer_experiment import TransferExperiment
from nntransfer.configs.experiment import Experiment
from nntransfer.configs import *
from orbit_transfer.configs import *

transfer_experiments = {}


class BaselineDataset(MNIST1DDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.train_shift = 30
        super().__init__(**kwargs)


class TeacherModel(MNIST1DModelConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type = "simple_fully_conv"
        self.kernel_size = 5
        self.padding = 2
        self.input_size = 40
        self.layers = 2
        self.channels = 15
        super().__init__(**kwargs)


class StudentModel(TeacherModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type = "simple_fc"
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


for forward, lr, gamma, softmax_temp in product(
    (
        "kd",
        "kd_match",
    ),
    (0.01, 0.001, 0.0001, 0.0005),
    list(np.linspace(0.1, 1.0, 10)),
    (0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0),
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
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"{forward}: gamma={gamma} T={softmax_temp} lr={lr}",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

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
      gamma    T    lr    0 train  Seen Shifts  0 validation  Unseen Shifts  \
name                                                                          
KD      0.3  2.0  0.01  80.416667         53.6          55.0           11.0   

      All Shifts  equiv  inv  id  
name                              
KD          39.9    NaN  NaN NaN  
                 gamma     T    lr  0 train  Seen Shifts  0 validation  \
name                                                                     
Direct Matching    0.1  10.0  0.01    83.25         54.7          58.0   

                 Unseen Shifts  All Shifts  equiv  inv  id  
name                                                        
Direct Matching           12.2        40.5    NaN  NaN NaN  
      gamma   T    lr    0 train  Seen Shifts  0 validation  Unseen Shifts  \
name                                                                         
RDL     0.6 NaN  0.01  82.444444         51.4         56.75           11.7   

      All Shifts  equiv  inv  id  
name                              
RDL         39.7    NaN  NaN NaN  
           gamma   T    lr    0 train  Seen Shifts  0 validation  \
name                                                               
Attention    0.6 NaN  0.01  83.333333         51.7          53.5   

           Unseen Shifts  All Shifts  equiv  inv  id  
name                                                  
Attention           12.3        40.0    NaN  NaN NaN  
              gamma   T    lr    0 train  Seen Shifts  0 validation  \
name                                                                  
Equivariance    1.0 NaN  0.01  66.333333         77.2          80.5   

              Unseen Shifts  All Shifts  equiv   inv   id  
name                                                       
Equivariance           76.5        77.3    0.1  10.0  1.0
"""