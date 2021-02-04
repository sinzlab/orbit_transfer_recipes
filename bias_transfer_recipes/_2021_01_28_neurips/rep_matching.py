from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

transfer_experiments = {}


class BaselineDataset(dataset.ImageDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "CIFAR100"
        self.input_size: int = 32
        self.add_corrupted_test: bool = False
        self.batch_size = 64
        super(BaselineDataset, self).__init__(**kwargs)


class BaselineModel(model.CIFAR100):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "resnet50"
        super(BaselineModel, self).__init__(**kwargs)


class BaselineTrainer(
    trainer.mixins.TransferMixin,
    trainer.mixins.NoiseAugmentationMixin,
    trainer.mixins.RepresentationMatchingMixin,
    trainer.Classification,
):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 200
        self.patience = 1000
        self.lr_milestones = (60, 120, 160)
        super(BaselineTrainer, self).__init__(**kwargs)

seed = 42

for rep_matching in (
    "representation matching bn-freeze",
    "no representation matching bn-freeze",
    "representation matching",
    "no representation matching",
):
    experiments = []
    layers = {
        # core output
        "flatten": "core",
    }
    transfer_settings = {
        "representation matching": [
            {
                "model": {
                    "get_intermediate_rep": layers,
                },
                "trainer": {
                    "representation_matching": {
                        "representations": list(layers.values()),
                        "criterion": "mse",
                        "combine_losses": "avg",
                        "second_noise_std": {(0, 0.5): 1.0},
                        "lambda": 1.0,
                        "only_for_clean": True,
                    },
                },
            },
            {"trainer": {"freeze_bn": False}},
        ],
        "representation matching bn-freeze": [
            {
                "model": {
                    "get_intermediate_rep": layers,
                },
                "trainer": {
                    "representation_matching": {
                        "representations": list(layers.values()),
                        "criterion": "mse",
                        "combine_losses": "avg",
                        "second_noise_std": {(0, 0.5): 1.0},
                        "lambda": 1.0,
                        "only_for_clean": True,
                    },
                },
            },
            {"trainer": {"freeze_bn": True}},
        ],
        "no representation matching bn-freeze": [{}, {"trainer": {"freeze_bn": True}}],
        "no representation matching": [{}, {"trainer": {"freeze_bn": False}}],
    }

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
                },
            ),
            seed=seed,
        )
    )

    # Step 2: Training on Clean
    experiments.append(
        Experiment(
            dataset=BaselineDataset(),
            model=BaselineModel(),
            trainer=BaselineTrainer(
                freeze=("core",),
                reset=("fc",),
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"Transfer noise augmented -> clean {rep_matching}",
            seed=seed,
        )
    ] = TransferExperiment(experiments, update=transfer_settings[rep_matching])

