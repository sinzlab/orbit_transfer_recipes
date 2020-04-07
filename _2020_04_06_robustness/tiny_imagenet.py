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
    ############# My config ###############
    # Clean baseline:
    experiments[Description(name="TinyImageNet", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(comment="TinyImageNet",
                                      apply_data_normalization=False,
                                      dataset_cls="TinyImageNet"),
        model=model.ModelConfig(comment="ResNet.TinyImageNet",
                                dataset_cls="TinyImageNet",
                                conv_stem_kernel_size=5),
        trainer=trainer.TrainerConfig(comment="", num_epochs=90, lr_milestones=(30, 60)),
        seed=seed)

    experiments[Description(name="TinyImageNet Noise Augmented",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="TinyImageNet",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="TinyImageNet"),
                                                     model=model.ModelConfig(comment="TinyImageNet",
                                                                             dataset_cls="TinyImageNet"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Noise Augmented",
                                                         lr_milestones=(30, 60),
                                                         num_epochs=90,
                                                         add_noise=True,
                                                         noise_snr=None,
                                                         noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1,
                                                                    0.26: 0.1, 0.38: 0.1, -1: 0.5}),
                                                     seed=seed)
    experiments[Description(name="TinyImageNet Transfer",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="TinyImageNet",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="TinyImageNet"),
                                                     model=model.ModelConfig(comment="TinyImageNet",
                                                                             dataset_cls="TinyImageNet"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Transfer + Reset",
                                                         lr_milestones=(30, 60),
                                                         num_epochs=90,
                                                         freeze=("core",),
                                                         reset_linear=True),
                                                     seed=seed)

    transfer_experiments[Description(name="TinyImageNet: Clean",
                                     seed=seed)] = TransferExperiment([experiments[Description(name="TinyImageNet",
                                                                                               seed=seed)],
                                                                       ])

    transfer_experiments[Description(name="TinyImageNet: Noise Augmented -> Transfer",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="TinyImageNet Noise Augmented",
                                 seed=seed)],
         experiments[Description(name="TinyImageNet Transfer",
                                 seed=seed)]
         ])

    ############# Shahd's config ###############
    # Clean baseline:
    experiments[Description(name="TinyImageNet (RMSprop)", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(comment="TinyImageNet",
                                      apply_data_normalization=False,
                                      dataset_cls="TinyImageNet"),
        model=model.ModelConfig(comment="ResNet.TinyImageNet",
                                dataset_cls="TinyImageNet",
                                conv_stem_kernel_size=5),
        trainer=trainer.TrainerConfig(comment="RMSprop",
                                      num_epochs=90,
                                      lr_milestones=(30, 60),
                                      lr=0.1,
                                      optimizer='RMSprop',
                                      lr_decay=0.1,
                                      weight_decay=5e-4,
                                      momentum=0.9
                                      ),
        seed=seed)

    experiments[Description(name="TinyImageNet Noise Augmented (RMSprop)",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="TinyImageNet",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="TinyImageNet"),
                                                     model=model.ModelConfig(comment="TinyImageNet",
                                                                             dataset_cls="TinyImageNet"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Noise Augmented (RMSprop)",
                                                         lr_milestones=(30, 60),
                                                         num_epochs=90,
                                                         lr=0.1,
                                                         optimizer='RMSprop',
                                                         lr_decay=0.1,
                                                         weight_decay=5e-4,
                                                         momentum=0.9,
                                                         add_noise=True,
                                                         noise_snr=None,
                                                         noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1,
                                                                    0.26: 0.1, 0.38: 0.1, -1: 0.5}),
                                                     seed=seed)
    experiments[Description(name="TinyImageNet Transfer (RMSprop)",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="TinyImageNet",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="TinyImageNet"),
                                                     model=model.ModelConfig(comment="TinyImageNet",
                                                                             dataset_cls="TinyImageNet"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Transfer + Reset (RMSprop)",
                                                         lr_milestones=(30, 60),
                                                         num_epochs=90,
                                                         lr=0.1,
                                                         optimizer='RMSprop',
                                                         lr_decay=0.1,
                                                         weight_decay=5e-4,
                                                         momentum=0.9,
                                                         freeze=("core",),
                                                         reset_linear=True),
                                                     seed=seed)

    experiments[Description(name="TinyImageNet Noise Augmented + Repr. Matching (Euclid, (0,0.5))(RMSprop)",
                            seed=seed)] = Experiment(dataset=dataset.DatasetConfig(comment="TinyImageNet",
                                                                                   apply_data_normalization=False,
                                                                                   dataset_cls="TinyImageNet"),
                                                     model=model.ModelConfig(comment="TinyImageNet",
                                                                             dataset_cls="TinyImageNet"),
                                                     trainer=trainer.TrainerConfig(
                                                         comment="Noise Augmented + Repr Matching (RMSprop)",
                                                         lr_milestones=(30, 60),
                                                         num_epochs=90,
                                                         lr=0.1,
                                                         optimizer='RMSprop',
                                                         lr_decay=0.1,
                                                         weight_decay=5e-4,
                                                         momentum=0.9,
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

    transfer_experiments[Description(name="TinyImageNet: Clean (RMSprop)",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="TinyImageNet (RMSprop)",
                                 seed=seed)],
         ])

    transfer_experiments[Description(name="TinyImageNet: Noise Augmented -> Transfer (RMSprop)",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="TinyImageNet Noise Augmented (RMSprop)",
                                 seed=seed)],
         experiments[Description(name="TinyImageNet Transfer (RMSprop)",
                                 seed=seed)]
         ])

    transfer_experiments[Description(name="TinyImageNet: Noise Augmented + Repr Matching -> Transfer (RMSprop)",
                                     seed=seed)] = TransferExperiment(
        [experiments[Description(name="TinyImageNet Noise Augmented + Repr. Matching (Euclid, (0,0.5))(RMSprop)",
                                 seed=seed)],
         experiments[Description(name="TinyImageNet Transfer (RMSprop)",
                                 seed=seed)]
         ])
