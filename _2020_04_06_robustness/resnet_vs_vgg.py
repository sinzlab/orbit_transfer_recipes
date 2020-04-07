from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

for seed in (8,
             13,
             42
             ):

    ########### ResNet ##############
    experiments[Description(name="ResNet: Clean", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(comment="",
                                      apply_data_normalization=False,
                                      dataset_cls="CIFAR100"),
        model=model.ModelConfig(comment="ResNet", dataset_cls="CIFAR100"),
        trainer=trainer.TrainerConfig(comment=""),
        seed=seed)

    experiments[Description(name="ResNet: Noise Augmented",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="CIFAR100"),
                                                     model=model.ModelConfig(comment="ResNet",
                                                                             dataset_cls="CIFAR100"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Noise Augmented",
                                                         add_noise=True,
                                                         noise_snr=None,
                                                         noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1,
                                                                    0.26: 0.1, 0.38: 0.1, -1: 0.5}),
                                                     seed=seed)

    experiments[Description(name="ResNet: Transfer",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="CIFAR100"),
                                                     model=model.ModelConfig(comment="ResNet",
                                                                             dataset_cls="CIFAR100"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Transfer + Reset",
                                                         freeze=("core",),
                                                         reset_linear=True),
                                                     seed=seed)

    experiments[Description(name="ResNet: Noise Augmented + Repr. Matching",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="CIFAR100"),
                                                     model=model.ModelConfig(comment="ResNet",
                                                                             dataset_cls="CIFAR100"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Noise Augmented + Repr Matching",
                                                         add_noise=True,
                                                         noise_snr=None,
                                                         noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1,
                                                                    0.26: 0.1, 0.38: 0.1, -1: 0.5},
                                                         representation_matching={"representation": "conv_rep",
                                                                                  "criterion": "mse",
                                                                                  "second_noise_std": {(0, 0.5): 1.0},
                                                                                  "lambda": 1.0}
                                                     ),
                                                     seed=seed)

    transfer_experiments[Description(name="ResNet: Clean",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="ResNet: Clean",
                                 seed=seed)],
         ])

    transfer_experiments[Description(name="ResNet: Noise Augmented -> Transfer",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="ResNet: Noise Augmented",
                                 seed=seed)],
         experiments[Description(name="ResNet: Transfer",
                                 seed=seed)]
         ])

    transfer_experiments[Description(name="ResNet: Noise Augmented + Repr Matching -> Transfer",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="ResNet: Noise Augmented + Repr. Matching",
                                 seed=seed)],
         experiments[Description(name="ResNet: Transfer",
                                 seed=seed)]
         ])

    ########## VGG ###################
    experiments[Description(name="VGG: Clean", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(comment="",
                                      apply_data_normalization=False,
                                      dataset_cls="CIFAR100"),
        model=model.ModelConfig(comment="VGG", dataset_cls="CIFAR100", cnn_builder="vgg", type="vgg19_bn"),
        trainer=trainer.TrainerConfig(comment=""),
        seed=seed)

    experiments[Description(name="VGG: Noise Augmented",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="CIFAR100"),
                                                     model=model.ModelConfig(comment="VGG", dataset_cls="CIFAR100",
                                                                             cnn_builder="vgg", type="vgg19_bn"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Noise Augmented",
                                                         add_noise=True,
                                                         noise_snr=None,
                                                         noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1,
                                                                    0.26: 0.1, 0.38: 0.1, -1: 0.5}),
                                                     seed=seed)

    experiments[Description(name="VGG: Transfer",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="CIFAR100"),
                                                     model=model.ModelConfig(comment="VGG", dataset_cls="CIFAR100",
                                                                             cnn_builder="vgg", type="vgg19_bn"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Transfer + Reset",
                                                         freeze=("core",),
                                                         reset_linear=True),
                                                     seed=seed)

    experiments[Description(name="VGG: Noise Augmented + Repr. Matching",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="CIFAR100"),
                                                     model=model.ModelConfig(comment="VGG", dataset_cls="CIFAR100",
                                                                             cnn_builder="vgg", type="vgg19_bn"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Noise Augmented + Repr Matching",
                                                         add_noise=True,
                                                         noise_snr=None,
                                                         noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1,
                                                                    0.26: 0.1, 0.38: 0.1, -1: 0.5},
                                                         representation_matching={"representation": "conv_rep",
                                                                                  "criterion": "mse",
                                                                                  "second_noise_std": {(0, 0.5): 1.0},
                                                                                  "lambda": 1.0}
                                                     ),
                                                     seed=seed)

    transfer_experiments[Description(name="VGG: Clean",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="VGG: Clean",
                                 seed=seed)],
         ])

    transfer_experiments[Description(name="VGG: Noise Augmented -> Transfer",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="VGG: Noise Augmented",
                                 seed=seed)],
         experiments[Description(name="VGG: Transfer",
                                 seed=seed)]
         ])

    transfer_experiments[Description(name="VGG: Noise Augmented + Repr Matching -> Transfer",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="VGG: Noise Augmented + Repr. Matching",
                                 seed=seed)],
         experiments[Description(name="VGG: Transfer",
                                 seed=seed)]
         ])

