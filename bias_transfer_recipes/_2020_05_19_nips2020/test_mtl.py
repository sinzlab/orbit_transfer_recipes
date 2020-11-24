from bias_transfer.configs.base import Description
from bias_transfer.configs.dataset import NeuralDatasetConfig, ImageDatasetConfig
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer


experiments = {}
train_cycler = "MTL_Cycler"
loss_weighing = True
scale_loss = False
ratios = [3]
opts = ["SGD"]
for ratio in ratios:
    for opt in opts:
        for seed in (1000,):
            for lr in [0.01]:
                optimizer_options = {"momentum": 0.9, "lr": lr, "weight_decay": 5e-4}
                dataset_name = "v1_crop_70_tinyimgnet_BW"
                model_name = "VGG19_MTL_classificationReadout_Conv_V1ReadoutLayer_{}_gamma_{}".format(
                    17, 0.5
                )
                trainer_name = "opt_{}_lr_{}_decay_0.6_trainCycler_{}_lossWeighing_{}_batchRatio_{}".format(
                    opt, lr, train_cycler, loss_weighing, ratio
                )
                experiments[
                    Description(
                        name=dataset_name + "_" + model_name + "_" + trainer_name,
                        seed=seed,
                    )
                ] = Experiment(
                    dataset=dataset.MTLDatasetsConfig(
                        {
                            "neural": NeuralDatasetConfig(
                                **{
                                    "comment": "",
                                    "dataset": "CSRF19_V1",
                                    "crop": 70,
                                    "seed": 1000,
                                    "subsample": 1,
                                }
                            ),
                            "img_classification": ImageDatasetConfig(
                                **{
                                    "comment": "",
                                    "dataset_cls": "TinyImageNet",
                                    "apply_grayscale": True,
                                    "add_stylized_test": False,
                                    "add_corrupted_test": True,
                                }
                            ),
                        },
                    ),
                    model=model.MTLModelConfig(
                        comment=model_name,
                        classification=True,
                        classification_readout_type="conv",
                        input_size=64,
                        num_classes=200,
                        pretrained=False,
                        vgg_type="vgg19",
                        v1_fine_tune=True,
                        v1_gamma_readout=0.5,
                        v1_model_layer=17,
                    ),
                    trainer=trainer.TrainerConfig(
                        comment=trainer_name,
                        optimizer=opt,
                        optimizer_options=optimizer_options,
                        train_cycler=train_cycler,
                        loss_weighing=loss_weighing,
                        loss_accum_batch_n=ratio + 1,
                        train_cycler_args={
                            "main_key": "img_classification",
                            "ratio": ratio,
                        },
                        max_iter=1,
                        scheduler="adaptive",
                        scheduler_options={"mtl": True},
                        mtl=True,
                        lr_decay_steps=10,
                        scale_loss=scale_loss,
                        patience=10,
                        lr_decay=0.6,
                        verbose=True,
                        loss_functions={
                            "img_classification": "CrossEntropyLoss",
                            "neural": "PoissonLoss",
                        },
                    ),
                    seed=seed,
                )

transfer_experiments = experiments
