from bias_transfer.configs.base import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

transfer_experiments = {}


class BaselineDataset(dataset.MNIST_IB):
    pass


class BaselineModel(model.MNISTIB):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.coreset_size = 200
        super(BaselineModel, self).__init__(**kwargs)


class BaselineTrainer(trainer.mixins.TransferMixin, trainer.Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.patience = 1000
        super(BaselineTrainer, self).__init__(**kwargs)


class BaselineRegressionTrainer(trainer.mixins.TransferMixin, trainer.Regression):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 100
        self.patience = 1000
        self.readout_name = "fc3"
        self.loss_functions = {"regression": "CircularDistanceLoss"}
        super(BaselineRegressionTrainer, self).__init__(**kwargs)


class DataGenerator(trainer.mixins.DataGenerationMixin, trainer.Classification):
    fn = "bias_transfer.trainer.transfer"


class DataGeneratorRegression(trainer.mixins.DataGenerationMixin, trainer.Regression):
    fn = "bias_transfer.trainer.regression_transfer"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.loss_functions = {"regression": "CircularDistanceLoss"}
        super(DataGeneratorRegression, self).__init__(**kwargs)


seed = 42
for bias in (
    ("clean", "color", "color_shuffle"),
    ("noise", "clean", "noise"),
    ("translation", "clean", "translation"),
    # ("rotation_regression", "clean", "rotation"),
):

    experiments = []
    # Step 1: Direct training on bias[1]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(
                dataset_cls="MNIST-IB",
                bias=bias[1],
            ),
            model=BaselineModel(
                bias=bias[1],
                input_channels=3 if "color" in bias[1] else 1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
            ),
            trainer=BaselineTrainer(
                comment=f"FashionMNIST Direct Training {bias[1]}",
            ),
            seed=seed,
        )
    )

    # Step 2: Test on bias[2]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(
                bias=bias[2],
            ),
            model=BaselineModel(
                bias=bias[2],
                input_channels=3 if "color" in bias[1] else 1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
            ),
            trainer=BaselineTrainer(
                comment=f"Test FashionMNIST-IB {bias[2]}",
                max_iter=0,
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"Direct Training A ({bias[0]}->{bias[1]};{bias[2]})",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

    experiments = []
    # Step 1: Direct training on bias[2]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(
                bias=bias[2],
            ),
            model=BaselineModel(
                bias=bias[2],
                input_channels=3 if "color" in bias[1] else 1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
            ),
            trainer=BaselineTrainer(
                comment=f"FashionMNIST-IB Direct Training {bias[2]}",
            ),
            seed=seed,
        )
    )

    # Step 2: Test on bias[1]
    experiments.append(
        Experiment(
            dataset=BaselineDataset(
                dataset_cls="MNIST-IB",
                bias=bias[1],
            ),
            model=BaselineModel(
                bias=bias[1],
                input_channels=3 if "color" in bias[1] else 1,
                type="lenet300-100" if bias[0] == "translation" else "lenet5",
            ),
            trainer=BaselineTrainer(
                comment=f"FashionMNIST Test {bias[1]}",
                max_iter=0
            ),
            seed=seed,
        )
    )

    transfer_experiments[
        Description(
            name=f"Direct Training B ({bias[2]};{bias[0]}->{bias[1]})",
            seed=seed,
        )
    ] = TransferExperiment(experiments)

