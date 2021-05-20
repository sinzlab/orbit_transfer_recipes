#!/usr/bin/python3


import argparse
import os
import stat
import subprocess
import time
from pathlib import Path


class SlurmJob:
    def __init__(
        self, recipe, experiment, name, time, gpu, num_gpus, memory, email, j=0
    ):
        self.recipe = recipe
        self.experiment = experiment
        self.name = f"{recipe}.{experiment}.{j}" if not name else name
        self.email = email
        self.time = time
        days, hours, minutes = list(
            map(int, [time.split("-")[0]] + time.split("-")[1].split(":"))
        )
        if "2080" in gpu:
            self.gpu = "gpu-2080ti-dev" if hours < 12 and days < 1 else "gpu-2080ti"
        else:
            self.gpu = "gpu-v100"
        self.gpu = "gpu-2080ti"
        self.num_gpus = num_gpus
        self.memory = memory

    @property
    def config_string(self):
        config_string = """
#SBATCH --job-name={name}             # Name of the job
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --cpus-per-task=2                       # Number of CPU cores per task
#SBATCH --nodes=1                               # Ensure that all cores are on one machine
#SBATCH --time={time}                      # Runtime in D-HH:MM
#SBATCH --partition={gpu}                  # Partition to submit to
#SBATCH --gres=gpu:{num_gpus}              # Number of requested GPUs
#SBATCH --mem-per-cpu={memory}             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=work/logs/{name}.%j.out      # File to which STDOUT will be written
#SBATCH --error=work/logs/{name}.%j.err       # File to which STDERR will be written
#SBATCH --mail-type=ALL                         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user={email}                # Email to which notifications will be sent
        """.format(
            name=self.name,
            time=self.time,
            gpu=self.gpu,
            num_gpus=self.num_gpus,
            memory=self.memory,
            email=self.email,
        )
        return config_string

    @property
    def cmd_string(self):
        cmd_string = """
scontrol show job $SLURM_JOB_ID  # print some info
singularity run --nv \
                --env-file .env \
                --env "CUDA_VISIBLE_DEVICES=0" \
                --no-home  \
                --bind $SCRATCH:$SCRATCH,$HOME/projects/:$HOME/projects/  \
                singularity_img.sif  \
                ./recipe_run.sh $SCRATCH {recipe} {experiment}
        """.format(
            recipe=self.recipe, experiment=self.experiment
        )
        return cmd_string

    def run(self):
        script = f"./work/{self.name}.sh"
        with open(script, "w") as f:
            f.write("#!/bin/bash \n \n" + self.config_string + "\n" + self.cmd_string)
        os.chmod(script, stat.S_IRWXU)
        print(subprocess.check_output("sbatch " + script, shell=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running pre-defined recipes or analysis"
    )
    parser.add_argument(
        "--njobs",
        dest="num_jobs",
        action="store",
        default=1,
        type=int,
        help="",
    )
    parser.add_argument(
        "--name",
        dest="name",
        action="store",
        default="",
        type=str,
        help="",
    )
    parser.add_argument(
        "--time",
        dest="time",
        action="store",
        default="0-00:00",
        type=str,
        help="time to complete each job. Specify in the following format: D-HH:MM",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store",
        default="2080",
        type=str,
        help="",
    )
    parser.add_argument(
        "--ngpus",
        dest="num_gpus",
        action="store",
        default=1,
        type=int,
        help="",
    )
    parser.add_argument(
        "--memory",
        dest="memory",
        action="store",
        default=3000,
        type=int,
        help="",
    )
    parser.add_argument(
        "--force-rebuild",
        dest="force_rebuild",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--recipe", dest="recipe", action="store", type=str, help="", required=True
    )
    parser.add_argument(
        "--experiment",
        dest="experiment",
        action="store",
        type=str,
        help="",
        default="",
        required=False,
    )
    parser.add_argument(
        "--email",
        dest="email",
        action="store",
        default=os.getenv("EMAIL"),
        type=str,
        help="",
    )

    args = parser.parse_args()
    if not Path("./singularity_img.sif").exists() or args.force_rebuild:
        print(
            subprocess.check_output(
                "singularity build --force --fakeroot singularity_img.sif ./singularity.def",
                shell=True,
            )
        )
    for j in range(args.num_jobs):
        job = SlurmJob(
            args.recipe,
            args.experiment,
            args.name,
            args.time,
            args.gpu,
            args.num_gpus,
            args.memory,
            args.email,
            j=j,
        )
        job.run()
        time.sleep(60)
