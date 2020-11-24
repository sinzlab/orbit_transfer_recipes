from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

for seed in (
    42,
    # 23,
    # 8
):
    experiments[
        Description(name="Sine Regression (Sin Activation)", seed=seed,)
    ] = Experiment(
        dataset=dataset.RegressionDatasetConfig(
            comment="Sine", dataset_cls="sine", batch_size=64, train_range=30
        ),
        model=model.RegressionModelConfig(comment="", layer_size=400, activation="sin"),
        trainer=trainer.RegressionTrainerConfig(
            comment="",
            max_iter=25000,
            lr_milestones=(5000, 15000),
            adaptive_lr=False,
            restore_best=True,
            early_stop=False,
            patience=10000000,
            show_epoch_progress=True,
        ),
        seed=seed,
    )

    experiments[Description(name="Sine Regression", seed=seed,)] = Experiment(
        dataset=dataset.RegressionDatasetConfig(
            comment="Sine", dataset_cls="sine", batch_size=64, train_range=30
        ),
        model=model.RegressionModelConfig(comment="", layer_size=400,),
        trainer=trainer.RegressionTrainerConfig(
            comment="",
            max_iter=25000,
            lr_milestones=(5000, 15000),
            adaptive_lr=False,
            restore_best=True,
            early_stop=False,
            patience=10000000,
            show_epoch_progress=True,
        ),
        seed=seed,
    )
    experiments[
        Description(
            name="Sine Regression Range 10 Transfer Core (Sin Activation)", seed=seed,
        )
    ] = Experiment(
        dataset=dataset.RegressionDatasetConfig(
            comment="Sine", dataset_cls="sine", batch_size=64, train_range=10
        ),
        model=model.RegressionModelConfig(comment="", layer_size=400, activation="sin"),
        trainer=trainer.RegressionTrainerConfig(
            comment="",
            max_iter=10000,
            lr_milestones=(5000, 7500),
            adaptive_lr=False,
            restore_best=True,
            early_stop=False,
            patience=10000000,
            show_epoch_progress=True,
            readout_name=-1,
            reset_linear=True,
            freeze=[str(i) for i in range(6)],
        ),
        seed=seed,
    )

    experiments[
        Description(
            name="Sine Regression Range 10 Transfer RDM", seed=seed,
        )
    ] = Experiment(
        dataset=dataset.RegressionDatasetConfig(
            comment="Sine", dataset_cls="sine", batch_size=64, train_range=10
        ),
        model=model.RegressionModelConfig(
            comment="", layer_size=400, activation="sigmoid", rdm_prediction={"lambda":1.0}
        ),
        trainer=trainer.RegressionTrainerConfig(
            comment="RDM Transfer",
            max_iter=1000,
            lr_milestones=(5000, 7500),
            adaptive_lr=False,
            restore_best=True,
            early_stop=False,
            patience=10000000,
            show_epoch_progress=False,
            rdm_transfer=True,
            rdm_prediction={"lambda":1.0},
        ),
        seed=seed,
    )
    experiments[
        Description(
            name="Sine Regression Range 10 Transfer RDM (Sin Activation)", seed=seed,
        )
    ] = Experiment(
        dataset=dataset.RegressionDatasetConfig(
            comment="Sine", dataset_cls="sine", batch_size=64, train_range=10
        ),
        model=model.RegressionModelConfig(
            comment="", layer_size=400, activation="sin", rdm_prediction={"lambda":1.0}
        ),
        trainer=trainer.RegressionTrainerConfig(
            comment="RDM Transfer",
            max_iter=1000,
            lr_milestones=(5000, 7500),
            adaptive_lr=False,
            restore_best=True,
            early_stop=False,
            patience=10000000,
            show_epoch_progress=False,
            rdm_transfer=True,
            rdm_prediction={"lambda":1.0},
        ),
        seed=seed,
    )
    experiments[
        Description(name="Sine Regression Range 10 Transfer Core", seed=seed,)
    ] = Experiment(
        dataset=dataset.RegressionDatasetConfig(
            comment="Sine", dataset_cls="sine", batch_size=64, train_range=10
        ),
        model=model.RegressionModelConfig(comment="", layer_size=400,),
        trainer=trainer.RegressionTrainerConfig(
            comment="",
            max_iter=10000,
            lr_milestones=(5000, 7500),
            adaptive_lr=False,
            restore_best=True,
            early_stop=False,
            patience=10000000,
            show_epoch_progress=True,
            readout_name=-1,
            reset_linear=True,
            freeze=[str(i) for i in range(6)],
        ),
        seed=seed,
    )
    experiments[
        Description(name="Sine Regression Range 10 (Sin Activation)", seed=seed,)
    ] = Experiment(
        dataset=dataset.RegressionDatasetConfig(
            comment="Sine", dataset_cls="sine", batch_size=64, train_range=10
        ),
        model=model.RegressionModelConfig(comment="", layer_size=400, activation="sin"),
        trainer=trainer.RegressionTrainerConfig(
            comment="",
            max_iter=10000,
            lr_milestones=(5000, 7500),
            adaptive_lr=False,
            restore_best=True,
            early_stop=False,
            patience=10000000,
            show_epoch_progress=True,
        ),
        seed=seed,
    )
    experiments[Description(name="Sine Regression Range 10", seed=seed,)] = Experiment(
        dataset=dataset.RegressionDatasetConfig(
            comment="Sine", dataset_cls="sine", batch_size=64, train_range=10
        ),
        model=model.RegressionModelConfig(comment="", layer_size=400,),
        trainer=trainer.RegressionTrainerConfig(
            comment="",
            max_iter=10000,
            lr_milestones=(5000, 7500),
            adaptive_lr=False,
            restore_best=True,
            early_stop=False,
            patience=10000000,
            show_epoch_progress=True,
        ),
        seed=seed,
    )

    transfer_experiments[
        Description(
            name="Sine Regression (Sin Activation) -> Transfer RDM",
            seed=seed,
        )
    ] = TransferExperiment(
        [
            experiments[
                Description(name="Sine Regression (Sin Activation)", seed=seed,)
            ],
            experiments[
                Description(
                    name="Sine Regression Range 10 Transfer RDM",
                    seed=seed,
                )
            ],
        ]
    )
    # transfer_experiments[
    #     Description(
    #         name="Sine Regression (Sin Activation) -> Transfer RDM (Sin Activation)",
    #         seed=seed,
    #     )
    # ] = TransferExperiment(
    #     [
    #         experiments[
    #             Description(name="Sine Regression (Sin Activation)", seed=seed,)
    #         ],
    #         experiments[
    #             Description(
    #                 name="Sine Regression Range 10 Transfer RDM (Sin Activation)",
    #                 seed=seed,
    #             )
    #         ],
    #     ]
    # )
    #
    # transfer_experiments[
    #     Description(name="Sine Regression (Sin Activation)", seed=seed,)
    # ] = TransferExperiment(
    #     [
    #         experiments[Description(name="Sine Regression Range 10 (Sin Activation)", seed=seed,)],
    #     ]
    # )
    # transfer_experiments[
    #     Description(name="Sine Regression (Sin Activation) -> Transfer Core", seed=seed,)
    # ] = TransferExperiment(
    #     [
    #         experiments[Description(name="Sine Regression (Sin Activation)", seed=seed,)],
    #         experiments[Description(name="Sine Regression Range 10 Transfer Core", seed=seed,)],
    #     ]
    # )
    # transfer_experiments[
    #     Description(name="Sine Regression -> Transfer Core", seed=seed,)
    # ] = TransferExperiment(
    #     [
    #         experiments[Description(name="Sine Regression", seed=seed,)],
    #         experiments[Description(name="Sine Regression Range 10 Transfer Core", seed=seed,)],
    #     ]
    # )
    # transfer_experiments[
    #     Description(name="Sine Regression", seed=seed,)
    # ] = TransferExperiment(
    #     [
    #         experiments[Description(name="Sine Regression Range 10", seed=seed,)],
    #     ]
    # )
    # transfer_experiments[
    #     Description(name="Sine Regression (Sin Activation) -> Transfer Core (Sin Activation)", seed=seed,)
    # ] = TransferExperiment(
    #     [
    #         experiments[Description(name="Sine Regression (Sin Activation)", seed=seed,)],
    #         experiments[Description(name="Sine Regression Range 10 Transfer Core (Sin Activation)", seed=seed,)],
    #     ]
    # )

