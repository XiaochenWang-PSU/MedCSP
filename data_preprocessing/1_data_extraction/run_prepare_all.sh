#!/bin/bash
set -euxo pipefail


python prepare_input.py --outcome=pretrain --T=48 --dt=1
cp -r ../data/processed/features/outcome=pretrain,T=48.0,dt=1.0 ../data/processed/features/benchmark,outcome=pretrain,T=48.0,dt=1.0
