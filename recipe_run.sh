#!/bin/bash

nvidia-smi
mkdir $SCRATCH/.cache
mkdir $SCRATCH/.config
ln -s $SCRATCH/.cache $HOME/.cache
ln -s $SCRATCH/.config $HOME/.config
python3 -m venv $1/env  --system-site-packages
. $1/env/bin/activate
python3 bias_transfer_recipes/main.py --recipe $2 "${@:3}"
