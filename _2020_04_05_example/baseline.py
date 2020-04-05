from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

seed = 42
experiments = {}
transfer_experiments = {}

experiments[Description(name="Noise Augmented",
                        seed=seed)] = Experiment(dataset=dataset.DatasetConfig(description="",
                                                                               dataset_cls="CIFAR10"),
                                                 model=model.ModelConfig(description="",
                                                                         dataset_cls="CIFAR10"),
                                                 trainer=trainer.TrainerConfig(
                                                     description="Noise Augmented",
                                                     num_epochs=2,
                                                     add_noise=True,
                                                     noise_snr=None,
                                                     noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1,
                                                                0.26: 0.1, 0.38: 0.1, -1: 0.5}),
                                                 seed=seed)
experiments[Description(name="Transfer",
                        seed=seed)] = Experiment(dataset=dataset.DatasetConfig(description="",
                                                                               dataset_cls="CIFAR10"),
                                                 model=model.ModelConfig(description="",
                                                                         dataset_cls="CIFAR10"),
                                                 trainer=trainer.TrainerConfig(
                                                     description="Transfer + Reset",
                                                     num_epochs=2,
                                                     freeze=("core",),
                                                     reset_linear=True),
                                                 seed=seed)

transfer_experiments[Description(name="Noise Augmented -> Transfer",
                                 seed=seed)] = TransferExperiment([experiments[Description(name="Noise Augmented",
                                                                                           seed=seed)],
                                                                   experiments[Description(name="Transfer",
                                                                                           seed=seed)]
                                                                   ])
