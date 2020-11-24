from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

matching_options = {
    "representation": "conv_rep",
    "criterion": "mse",
    "second_noise_std": {(0, 0.5): 1.0},
    "lambda": 1.0,
}
noise_type = {
    "add_noise": True,
    "noise_snr": None,
    "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5,},
}

dataset_cls = "CIFAR10"
seed = 42
sparsity = 80

experiments[
    Description(
        name="CIFAR10: Clean + Pruning {} (Adam)".format(sparsity), seed=seed
    )
] = Experiment(
    dataset=dataset.ImageDatasetConfig(
        comment=dataset_cls,
        dataset_cls=dataset_cls,
        apply_data_normalization=True,
        add_corrupted_test=False,
        batch_size=128,
    ),
    model=model.ClassificationModelConfig(
        comment=dataset_cls,
        dataset_cls=dataset_cls,
        type="resnet18",
        conv_stem_kernel_size=3,
    ),
    trainer=trainer.TrainerConfig(
        optimizer="Adam",
        optimizer_options={
            "amsgrad": True,
            "lr": 0.0003,
            "weight_decay": 5e-4,
        },
        lr_decay=0.8,
        lr_warmup=10,
        max_iter=80,
        lr_milestones=(50, 65),
        lottery_ticket={
            "rounds": 2,
            "round_length": 1,
            "percent_to_prune": sparsity,
            "pruning": True,
            "reinit": False,
            "global_pruning": True,
        },
        adaptive_lr=False,
        restore_best=False,
        early_stop=False,
        patience=1000,
        comment="Clean lottery-ticket Adam",
        verbose=True,
        noise_test={}
    ),
    seed=seed,
)

# experiments[Description(name="CIFAR10: Transfer", seed=42)] = Experiment(
#     dataset=dataset.ImageDatasetConfig(
#         comment="CIFAR10",
#         dataset_cls="CIFAR10",
#         apply_data_normalization=True,
#         add_corrupted_test=False,
#     ),
#     model=model.ClassificationModelConfig(
#         comment="CIFAR10",
#         dataset_cls="CIFAR10",
#         type="resnet18",
#     ),
#     trainer=trainer.TrainerConfig(
#         comment="", max_iter=3, verbose=False, freeze=("core",), reset_linear=True
#     ),
#     seed=42,
# )
# experiments[Description(name="neural_prediction_test", seed=42)] = Experiment(
#     dataset=dataset.NeuralDatasetConfig(comment=""),
#     model=model.NeuralModelConfig(comment=""),
#     trainer=trainer.TrainerConfig(
#         comment="",
#         loss_accum_batch_n=None,
#         neural_prediction=True,
#         patience=5,
#         lr_milestones=None,
#         adaptive_lr=True,
#         lr=0.005,
#         threshold=1e-6,
#         lr_decay=0.3,
#         max_iter=1,
#     ),
#     seed=42,
# )
#
# transfer_experiments[Description(name="CIFAR10", seed=42)] = TransferExperiment(
#     [
#         experiments[Description(name="CIFAR10", seed=42)],
#         # experiments[Description(name="CIFAR10: Transfer", seed=42)],
#     ]
# )

transfer_experiments = experiments

# transfer_experiments[
#     Description(name="Neural prediction", seed=42)
# ] = TransferExperiment(
#     [experiments[Description(name="neural_prediction_test", seed=42)],]
# )
