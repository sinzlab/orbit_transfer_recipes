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
         lr  weight_decay  hidden_dim    0 train  Seen Shifts  0 validation  \
name                                                                          
ce    0.001        0.0001       600.0  83.333333         50.4         54.25   

      Unseen Shifts  All Shifts  gamma   T  equiv  inv  id  
name                                                        
ce             11.1        40.0    NaN NaN    NaN  NaN NaN  
         lr  weight_decay  hidden_dim    0 train  Seen Shifts  0 validation  \
name                                                                          
KD    0.001        0.0001       600.0  83.333333         52.3         52.25   

      Unseen Shifts  All Shifts  gamma    T  equiv  inv  id  
name                                                         
KD             13.5        39.0    0.6  1.0    NaN  NaN NaN  
                    lr  weight_decay  hidden_dim    0 train  Seen Shifts  \
name                                                                       
Direct Matching  0.001  1.000000e-08         NaN  83.333333         97.2   

                 0 validation  Unseen Shifts  All Shifts  gamma    T  equiv  \
name                                                                          
Direct Matching          96.0           90.6        94.5    0.4  5.0    NaN   

                 inv  id  
name                      
Direct Matching  NaN NaN  
          lr  weight_decay  hidden_dim    0 train  Seen Shifts  0 validation  \
name                                                                           
RDL   0.0001           NaN         NaN  82.666667         55.6         51.25   

      Unseen Shifts  All Shifts  gamma   T  equiv  inv  id  
name                                                        
RDL            14.4        42.6    0.9 NaN    NaN  NaN NaN  
         lr  weight_decay  hidden_dim    0 train  Seen Shifts  0 validation  \
name                                                                          
cka   0.001           NaN         NaN  83.333333         56.9          57.0   

      Unseen Shifts  All Shifts  gamma   T  equiv  inv  id  
name                                                        
cka            17.6        45.7    0.9 NaN    NaN  NaN NaN  
              lr  weight_decay  hidden_dim    0 train  Seen Shifts  \
name                                                                 
Attention  0.001           NaN         NaN  83.333333         55.8   

           0 validation  Unseen Shifts  All Shifts  gamma   T  equiv  inv  id  
name                                                                           
Attention          60.0           12.0        43.0    0.9 NaN    NaN  NaN NaN  
         lr  weight_decay  hidden_dim  0 train  Seen Shifts  0 validation  \
name                                                                        
Orbit  0.01           NaN         NaN     79.5         95.4         95.75   

       Unseen Shifts  All Shifts  gamma   T  equiv   inv    id  
name                                                            
Orbit           95.7        95.7    1.0 NaN    0.1  10.0  10.0  

"""