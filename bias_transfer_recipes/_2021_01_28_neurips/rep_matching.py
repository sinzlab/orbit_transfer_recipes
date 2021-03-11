from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

transfer_experiments = {}


class BaselineDataset(dataset.TinyImageNet):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "TinyImageNet"
        self.add_corrupted_test: bool = False
        self.apply_grayscale: bool = True
        self.batch_size = 128
        super(BaselineDataset, self).__init__(**kwargs)


class BaselineModel(model.TinyImageNet):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.input_channels = 1
        self.type: str = "vgg19-bn"
        # self.readout_type: str = "conv"
        super(BaselineModel, self).__init__(**kwargs)


class BaselineTrainer(
    trainer.mixins.TransferMixin,
    trainer.mixins.NoiseAugmentationMixin,
    trainer.mixins.RepresentationMatchingMixin,
    trainer.mixins.RepresentationMonitorMixin,
    trainer.Classification,
):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.optimizer = "SGD"
        self.optimizer_options = {"momentum": 0.8, "lr": 0.1, "weight_decay": 5e-5}
        self.lr_decay = 0.3
        self.patience = 10
        self.lr_decay_steps = 5
        self.threshold = 1e-6
        self.scheduler = "adaptive"
        super(BaselineTrainer, self).__init__(**kwargs)


seed = 42

vgg_layers = [
    "features.0.weight",
    "features.0.bias",
    "features.1.weight",
    "features.1.bias",
    "features.3.weight",
    "features.3.bias",
    "features.4.weight",
    "features.4.bias",
    "features.7.weight",
    "features.7.bias",
    "features.8.weight",
    "features.8.bias",
    "features.10.weight",
    "features.10.bias",
    "features.11.weight",
    "features.11.bias",
    "features.14.weight",
    "features.14.bias",
    "features.15.weight",
    "features.15.bias",
    "features.17.weight",
    "features.17.bias",
    "features.18.weight",
    "features.18.bias",
    "features.20.weight",
    "features.20.bias",
    "features.21.weight",
    "features.21.bias",
    "features.23.weight",
    "features.23.bias",
    "features.24.weight",
    "features.24.bias",
    "features.27.weight",
    "features.27.bias",
    "features.28.weight",
    "features.28.bias",
    "features.30.weight",
    "features.30.bias",
    "features.31.weight",
    "features.31.bias",
    "features.33.weight",
    "features.33.bias",
    "features.34.weight",
    "features.34.bias",
    "features.36.weight",
    "features.36.bias",
    "features.37.weight",
    "features.37.bias",
    "features.40.weight",
    "features.40.bias",
    "features.41.weight",
    "features.41.bias",
    "features.43.weight",
    "features.43.bias",
    "features.44.weight",
    "features.44.bias",
    "features.46.weight",
    "features.46.bias",
    "features.47.weight",
    "features.47.bias",
    "features.49.weight",
    "features.49.bias",
    "features.50.weight",
    "features.50.bias",
    "classifier.0.weight",
    "classifier.0.bias",
    "classifier.3.weight",
    "classifier.3.bias",
    "classifier.6.weight",
    "classifier.6.bias",
]

for rep_matching in (
    # "representation matching bn-freeze",
    "representation matching bn-freeze cross-noise",
    "representation matching bn-freeze cross-noise cosine",
    "no representation matching bn-freeze",
    # "representation matching cross-noise",
    # "representation matching",
    # "no representation matching",
):
    for layers in (
        # {
        #     "features.0": "conv-1-1",
        #     "features.3": "conv-1-2",
        #     "features.7": "conv-2-1",
        #     "features.10": "conv-2-2",
        #     "features.14": "conv-3-1_and_before",
        # },
        {"features.14": "conv-3-1"},
        {"features.14": "conv-3-1_extra_layer"},
        # {"features.27": "conv-4-1"},
        # {"features.40": "conv-5-1"},
        # {"features.49": "core"},
    ):
        experiments = []
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
                            "extra_layer": ("extra_layer" in list(layers.values())[-1]),
                        },
                    },
                },
                {"trainer": {"freeze_bn": False}},
            ],
            "representation matching cross-noise": [
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
                            "only_for_clean": False,
                            "extra_layer": ("extra_layer" in list(layers.values())[-1]),
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
                            "extra_layer": ("extra_layer" in list(layers.values())[-1]),
                        },
                    },
                },
                {"trainer": {"freeze_bn": True}},
            ],
            "representation matching bn-freeze cross-noise cosine": [
                {
                    "model": {
                        "get_intermediate_rep": layers,
                    },
                    "trainer": {
                        "representation_monitor": {
                            "representations": list(layers.values()),
                        },
                        "representation_matching": {
                            "representations": list(layers.values()),
                            "criterion": "cosine",
                            "combine_losses": "avg",
                            "second_noise_std": {(0, 0.5): 1.0},
                            "lambda": 1.0,
                            "only_for_clean": False,
                            "extra_layer": ("extra_layer" in list(layers.values())[-1]),
                        },
                    },
                },
                {"trainer": {"freeze_bn": True}},
            ],
            "representation matching bn-freeze cross-noise": [
                {
                    "model": {
                        "get_intermediate_rep": layers,
                    },
                    "trainer": {
                        "representation_monitor": {
                            "representations": list(layers.values()),
                        },
                        "representation_matching": {
                            "representations": list(layers.values()),
                            "criterion": "mse",
                            "combine_losses": "avg",
                            "second_noise_std": {(0, 0.5): 1.0},
                            "lambda": 1.0,
                            "only_for_clean": False,
                            "extra_layer": ("extra_layer" in list(layers.values())[-1]),
                        },
                    },
                },
                {"trainer": {"freeze_bn": True}},
            ],
            "no representation matching bn-freeze": [
                {
                    "model": {
                        "get_intermediate_rep": layers,
                    },
                    "trainer": {
                        "representation_monitor": {
                            "representations": list(layers.values()),
                        },
                    },
                },
                {"trainer": {"freeze_bn": True}},
            ],
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
        layer_key = list(layers.keys())[-1]
        layer_index = len(vgg_layers) - 1
        while layer_key not in vgg_layers[layer_index]:
            layer_index -= 1
            if layer_index == 0:
                break
        layer_index += 1
        to_freeze = vgg_layers[:layer_index]
        to_reset = vgg_layers[layer_index:]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(),
                model=BaselineModel(),
                trainer=BaselineTrainer(
                    freeze=to_freeze,
                    reset=to_reset,
                ),
                seed=seed,
            )
        )
        transfer_experiments[
            Description(
                name=f"Transfer noise augmented {rep_matching} {list(layers.values())[-1]}",
                seed=seed,
            )
        ] = TransferExperiment(
            experiments[:1], update=transfer_settings[rep_matching][:1]
        )

        transfer_experiments[
            Description(
                name=f"Transfer noise augmented -> clean ({rep_matching}, {list(layers.values())[-1]})",
                seed=seed,
            )
        ] = TransferExperiment(experiments, update=transfer_settings[rep_matching])

#
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
            },
        ),
        seed=seed,
    )
)

transfer_experiments[
    Description(
        name=f"Noise augmented",
        seed=seed,
    )
] = TransferExperiment(
    experiments[:1], update=transfer_settings["no representation matching"][:1]
)
