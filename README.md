# bias_transfer_recipes
This repository serves as a template of experiment recipes for the `bias_transfer` framework.
Any interaction with the `bias_transfer` framework should be performed through this repository. 
This can be done by either executing the `main.py`-script or through on of the notebooks. 

## Concept and Conventions
This repository should serve as a collection of all experiments a user ever ran with the `bias_transfer` framework. 
Furthermore, it should also contain any analysis and experimentation done, by keeping a collection of notebooks.
All of this should be 100% reproducible at any point in time, independent of changes performed at later times in any of the depending code-bases.
This reproducability is achieved by keeping a strict structure for all experiments and saving the id of the git-commits an experiment is supposed to be run against.
When executing an experiment, this framework will checkout, install and import the specified version of each depending git repo to execute the experiment in exactly the same way every time.

### Recipe
A folder containing all experiment definitions belonging to a specific project (stuff that should be kept within the same schema). 
The folder should abide by the following naming scheme:
```
_yyyy_mm_dd_recipe_name/
```
The most important file in every recipe folder is the `__commits.json`. 
This file should specify a specific commit-hash for each repository this project depends on. 
The specification could either be done for the whole recipe (by giving `default`-values), for each experiment individually or a combination of both.
An example could look like this:
```
{
  "default": {
    "bias_transfer": "fc12b41b68233f1829004180412bfbdb1946845c",
    "nnfabrik": "a63ba9c5a3ad1fa18a80f67a124f0af3b5e053d3",
    "nnvision": "69d02beea2f56470f7f19243527f77745ab4d5d9",
    "mlutils": "3b0958aa7d95b4978c801694a361be67c1049f06"
  },
  "baseline": {
    "bias_transfer": "d8750eb4bb84e6090da506a4922aabb6ff012296",
    "nnfabrik": "",
    "nnvision": "",
    "mlutils": ""
  }
}
```

### Experiment
An experiment is a normal python file (with a name corresponding to the entries in `__commits.json`) defining multiple instances of the `Experiment`-class from `bias_transfer.configs.experiment`.
Such an experiment instance is defined by the `dataset`,`model` and `trainer` config as well as the `seed`. 
```python
experiments[Description(name="Noise Augmented",
                        seed=seed)] = Experiment(dataset=dataset.DatasetConfig(description="",
                                                                               dataset_cls="CIFAR10"),
                                                 model=model.ModelConfig(description="",
                                                                         dataset_cls="CIFAR10"),
                                                 trainer=trainer.TrainerConfig(
                                                     description="Noise Augmented",
                                                     num_epochs=2,
                                                     add_noise=True,
                                                     noise_snr=None,
                                                     noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1,
                                                                0.26: 0.1, 0.38: 0.1, -1: 0.5}),
                                                 seed=seed)
```
It is important to save all experiments in the `experiments` dictionary, identifying it by `name` and `seed`. 

Experiments can also be combined to `TransferExperiment`s, which will lead to them being executed sequentially with the model from the previous run as initialization.
```python
transfer_experiments[Description(name="Noise Augmented -> Transfer",
                                 seed=seed)] = TransferExperiment([experiments[Description(name="Noise Augmented",
                                                                                           seed=seed)],
                                                                   experiments[Description(name="Transfer",
                                                                                           seed=seed)]
                                                                   ])
```
Again, these experiments are saved in the `transfer_experiments` dictionary, which will be used when the experiment is executed.
Note that it is also possible to add single stage experiments to this dictionary, to allow for maximum flexibility.

## Setup
To execute scripts and notebooks from this repository, it is necessary to structure the environment in a specific way.

### Folder Structure
The parent folder of `bias_transfer_recipes` is expected to fit the following structure:
```
   bias_transfer_recipes/
   bias_transfer/
   mlutils/
   nnfabrik/
   nnvision/
   work/data/
   Dockerfile
```
That means it is expected to clone the following repositories into the same folder as `bias_transfer_recipes`:
- https://github.com/sinzlab/bias_transfer 
- https://github.com/sinzlab/ml-utils
- https://github.com/sinzlab/nnfabrik
- https://github.com/sinzlab/nnvision 

Additionally, there should be a `work`-folder located in the same parent directory. 
This folder should contain the `data`-directory, or a link to where the data is stored. 

### Dockerfile
The parent folder should also contain a `Dockerfile`, which should be based on the following example:
```
FROM sinzlab/pytorch:v1.3.1-cuda10.1-dj0.12.4

# install third-party libraries
# needed for vim extension:
RUN apt-get update && apt-get install -y \
                                  nodejs \
                                  npm
RUN jupyter labextension install jupyterlab_vim
RUN pip install checkout_code
RUN pip install requests
RUN pip install imageio

RUN mkdir -p /recipes
RUN mkdir -p /work

# preparing repositories
RUN mkdir -p /src/bias_transfer
RUN mkdir -p /src/nnfabrik
RUN mkdir -p /src/mlutils
RUN mkdir -p /src/nnvision
COPY bias_transfer /src/bias_transfer
COPY nnfabrik /src/nnfabrik
COPY mlutils /src/mlutils
COPY nnfabrik /src/nnvision
```
### .env File
Everything in this repository will be running in a docker container via `docker-compose`. 
This expects an `.env`-file to be added to the main folder of this repository.
Therfore it is curcial to add (but keep out of the git-repository) a file like this:
```
MINIO_ENDPOINT=s3.amazonaws.com
MINIO_ACCESS_KEY=<minio_access_key>
MINIO_SECRET_KEY=<minio_secret_key>
DJ_HOST=sinzlab.chlkmukhxp6i.eu-central-1.rds.amazonaws.com
DJ_USER=<dj_use_name>
DJ_PASS=<dj_password>
JUPYTER_PASSWORD=<jupyter_password>
USER=<user_full_name>
EMAIL=<user_email>
AFFILIATION=sinzlab
```
## Execution
### Commandline Interface
In general, to run any experiment, it is recommended to run it via the commandline interface that is provided by `main.py`.
For this purpose, simply execute the `server_job` service with `docker-compose`:
```
$ docker-compose run --name <user>_prod_GPU<gpu_index> -d -e NVIDIA_VISIBLE_DEVICES=<gpu_index> server_job main.py --recipe <recipe> --experiment <experiment> --schema <schema>
```
Running the example experiment from this repository would look something like this:
```
$ docker-compose run --name anix_prod_GPU1 -d -e NVIDIA_VISIBLE_DEVICES=1 server_job main.py  --recipe _2020_04_05_example --experiment baseline --schema anix_nnfabrik_example_schema
```
Alternatively, it is also possible to start the jupyter interface (see below) and attach an interactive shell to start the experiment manually:
```
$ docker exec -i -t <user>_prod_GPU<gpu_index> /bin/bash
$ python3 main.py --recipe <recipe> --experiment <experiment> --schema <schema>
```
### Jupyter Interface
To run jupyter notebooks with this setup, simply log into any GPU-machine, select a gpu (=`<gpu_index>`) and run the following command:
```
$ docker-compose run -p 127.0.0.1:<port_on_server>:8888 --name <user>_dev_GPU<gpu_index> -d -e NVIDIA_VISIBLE_DEVICES=<gpu-index> server_develop
```
To connect to this notebook from a local machine, it is necessary to forward the selected port:
```
$ ssh -f -L <port_on_local>:0.0.0.0:<port_on_server> -N <server>
```
e.g. in my case it would look like
```
$ docker-compose run -p 127.0.0.1:10140:8888 --name anix_dev_GPU1 -d -e NVIDIA_VISIBLE_DEVICES=1 server_develop
$ ssh -f -L 8888:0.0.0.0:10140 -N gpu-server
```
Now the notebook should be accessible via `http://localhost:<port_on_local>/lab?` and ready to run. Simply follow the example-notebook for a starting point.
