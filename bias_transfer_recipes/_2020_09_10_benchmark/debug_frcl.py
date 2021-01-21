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
        self.type = "lenet300-100-frcl"
        self.coreset_size = 200
        super(BaselineModel, self).__init__(**kwargs)


class BaselineTrainer(trainer.mixins.TransferMixin, trainer.Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 1
        self.patience = 1000
        super(BaselineTrainer, self).__init__(**kwargs)


class BaselineRegressionTrainer(trainer.mixins.TransferMixin, trainer.Regression):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.max_iter = 1
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
for bias in (("color_easy", "clean_shuffle", "color_easy"),):
    for alpha in (1.0,):
        experiments = []
        transfer = "FRCL"
        softmax_temp = 1.0
        transfer_settings = {
            "FRCL": [
                {
                    "trainer": {
                        "extract_coreset": {
                            "method": "random",
                            "size": 200,
                            "save_in_model": True,
                        },
                    },
                },
                {
                    "trainer": {
                        "reset": "all",
                        "regularization": {"regularizer": "FRCL",},
                    },
                },
                {
                    "trainer": {
                        "extract_coreset": {
                            "method": "frcl",
                            "initial_method": "random",
                            "size": 200,
                            "save_in_model": True,
                            "max_iter": 10,
                        },
                        "reset_for_new_task": True,
                    },
                },
                {
                    "trainer": {
                        "extract_coreset": {
                            "method": "random",
                            "size": 200,
                            "save_in_model": True,
                        },
                    },
                },
                {
                    "trainer": {
                        "reset": "all",
                        "regularization": {"regularizer": "FRCL",},
                    },
                },
                {
                    "trainer": {
                        "regularization": {"regularizer": "FRCL",},
                    },
                },
            ],
        }
        if transfer == "KnowledgeDistillation" and "regression" in bias[0]:
            continue
        trainer_config_cls = (
            trainer.Regression if "regression" in bias[0] else BaselineTrainer
        )
        transfer_config_cls = (
            DataGeneratorRegression if "regression" in bias[0] else DataGenerator
        )

        # Step 1 extract initial coreset on bias[0]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(bias=bias[0], shuffle=False, valid_size=0.0,),
                model=BaselineModel(bias=bias[0],),
                trainer=transfer_config_cls(
                    comment=f"FashionMNIST Data Generation ({transfer}) {bias[0]}",
                ),
                seed=seed,
            )
        )

        # Step 2 FRCL training on bias[0]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(bias=bias[0],),
                model=BaselineModel(bias=bias[0],),
                trainer=trainer_config_cls(comment=f"FashionMNIST-IB {bias[0]}",),
                seed=seed,
            )
        )

        # Step 3: extract coreset with trained model on bias[0] and reset for new task
        experiments.append(
            Experiment(
                dataset=BaselineDataset(bias=bias[0], shuffle=False, valid_size=0.0,),
                model=BaselineModel(bias=bias[0],),
                trainer=transfer_config_cls(
                    comment=f" FashionMNIST Data Generation ({transfer}) {bias[0]}",
                ),
                seed=seed,
            )
        )

        # Step 4: extract initial coreset on bias[1]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(bias=bias[1], shuffle=False, valid_size=0.0,
                                        convert_to_rgb=("color" in bias[0]),
                                        ),
                model=BaselineModel(bias=bias[1],
                                    input_channels=3 if "color" in bias[0] else 1,
                                    ),
                trainer=transfer_config_cls(
                    comment=f" FashionMNIST Data Generation ({transfer}) {bias[1]}",
                ),
                seed=seed,
            )
        )

        # Step 5: FRCL training on bias[1]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(
                    dataset_cls="MNIST-IB",
                    bias=bias[1],
                    convert_to_rgb=("color" in bias[0]),
                ),
                model=BaselineModel(
                    bias=bias[1],
                    input_channels=3 if "color" in bias[0] else 1,
                ),
                trainer=BaselineTrainer(
                    comment=f"FashionMNIST Transfer ({transfer}) {bias[1]}",
                ),
                seed=seed,
            )
        )

        # step 6: Test on bias[2]
        experiments.append(
            Experiment(
                dataset=BaselineDataset(bias=bias[2],),
                model=BaselineModel(
                    bias=bias[2],
                    input_channels=3 if "color" in bias[0] else 1,
                ),
                trainer=BaselineTrainer(
                    comment=f"Test FashionMNIST-IB {bias[2]}", max_iter=0,
                ),
                seed=seed,
            )
        )

        transfer_experiments[
            Description(
                name=f"Transfer {transfer}: {alpha} ({bias[0]}->{bias[1]};{bias[2]})",
                seed=seed,
            )
        ] = TransferExperiment(experiments, update=transfer_settings[transfer])
