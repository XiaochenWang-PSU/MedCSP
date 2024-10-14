#!/bin/bash
set -euxo pipefail

export PYTHONPATH="../../FIDDLE/"
DATAPATH=$(python -c "import yaml;print(yaml.full_load(open('../config.yaml'))['data_path']);")
mkdir -p log




python -m FIDDLE.run \
    --data_fname="$DATAPATH/features/outcome=pretrain,T=48.0,dt=1.0/input_data.p" \
    --output_dir="$DATAPATH/features/outcome=pretrain,T=48.0,dt=1.0/" \
    --population="$DATAPATH/population/pretrain_48.0h.csv" \
    --T=48.0 \
    --dt=1.0 \
    --theta_1=0.05 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    > >(tee 'log/benchmark,outcome=mortality,T=48.0,dt=1.0.out') \
    2> >(tee 'log/benchmark,outcome=mortality,T=48.0,dt=1.0.err' >&2)
