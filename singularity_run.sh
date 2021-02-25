#!/bin/bash
mkdir -p /tmp/djcache
singularity $1 --nv \
                 --env-file .env \
                 --env "CUDA_VISIBLE_DEVICES=$2" \
                 --bind /tmp/djcache:/cache,/var/sinz-shared:/var/sinz-shared \
                 singularity_img.sif \
                 "$3" \
                 ./jupyter_run.sh
#                 python3 -m venv ./env  --system-site-packages && \
#                 sleep 5 && \
#                 pip freeze && \
#                 . ./env/bin/activate && \
#                 pip freeze && \
#                 "${@:4}"

# ./singularity_run.sh run 1 python3 bias_transfer_recipes/main.py --recipe _2020_09_10_benchmark --experiment test
# ./singularity_run.sh run 1 jupyter lab