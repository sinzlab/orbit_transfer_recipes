from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

# for seed in (8,
#              # 13,
#              # 42
#              ):
#     ############# My config ###############
#     # Clean baseline:
#     experiments[Description(name="TinyImageNet", seed=seed)] = Experiment(
#         dataset=dataset.DatasetConfig(comment="TinyImageNet",
#                                       apply_data_normalization=False,
#                                       dataset_cls="TinyImageNet",
#                                       add_corrupted_test=True),
#         model=model.ModelConfig(comment="ResNet.TinyImageNet",
#                                 dataset_cls="TinyImageNet",
#                                 conv_stem_kernel_size=5),
#         trainer=trainer.TrainerConfig(comment="", num_epochs=1, lr_milestones=(30, 60)),
#         seed=seed)
#
#
#     transfer_experiments[Description(name="TinyImageNet: Clean",
#                                      seed=seed)] = TransferExperiment([experiments[Description(name="TinyImageNet",
#                                                                                                seed=seed)],
#                                                                        ])

experiments[Description(name="CIFAR10: Clean", seed=42)] = Experiment(
        dataset=dataset.DatasetConfig(comment="CIFAR10", dataset_cls="CIFAR10", apply_data_normalization=False,
                                      add_corrupted_test=False),
        model=model.ModelConfig(comment="CIFAR10", dataset_cls="CIFAR10"),
        trainer=trainer.TrainerConfig(comment="",
                                      num_epochs=1),
        seed=42)


experiments[Description(name="CIFAR10: Clean", seed=42)] = Experiment(
        dataset=dataset.DatasetConfig(comment="CIFAR10", dataset_cls="CIFAR10", apply_data_normalization=False,
                                      add_corrupted_test=False),
        model=model.ModelConfig(comment="CIFAR10", dataset_cls="CIFAR10"),
        trainer=trainer.TrainerConfig(comment="",
                                      num_epochs=1,
                                      rdm_transfer=True),
        seed=42)
transfer_experiments[Description(name="CIFAR10: Clean",
                                 seed=42)] = TransferExperiment([experiments[Description(name="CIFAR10: Clean",
                                                                                           seed=42)],
                                                                       ])

