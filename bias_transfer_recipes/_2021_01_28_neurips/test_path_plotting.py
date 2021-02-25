from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

transfer_experiments = {}


class BaselineDataset(dataset.MNIST):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.batch_size = 256
        super(BaselineDataset, self).__init__(**kwargs)


class BaselineModel(model.MNIST):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        super(BaselineModel, self).__init__(**kwargs)


class BaselineTrainer(
    trainer.mixins.TransferMixin,
    trainer.mixins.NoiseAugmentationMixin,
    trainer.Classification,
):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 10
        self.patience = 1000
        self.lr_milestones = (3, 6)
        super(BaselineTrainer, self).__init__(**kwargs)


for seed, noise in ((42, True),
                    # (42, False), (13, True), (13, False)
                    ):
    experiments = []

    # Step 1: Training on Noise
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=BaselineModel(),
            trainer=BaselineTrainer(
                comment=f"Training ",
                noise_std={
                    0.08: 0.1,
                    0.12: 0.1,
                    0.18: 0.1,
                    0.26: 0.1,
                    0.38: 0.1,
                    -1: 0.5,
                }
                if noise
                else {},
                # chkpt_options={
                #     # "save_every_n": 1,
                #     # "keep_best_n": 1,
                #     # "keep_last_n": 1000,
                #     # "keep_selection": (),
                # },
                # keep_checkpoints=True
            ),
            seed=seed,
        )
    )
    transfer_experiments[
        Description(
            name=f"Train " + ("noisy" if noise else ""),
            seed=seed,
        )
    ] = TransferExperiment(experiments)
