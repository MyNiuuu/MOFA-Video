#!/bin/bash
work_path=$(dirname $0)
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=2 --node_rank=$1 \
    --master_addr="192.168.1.1" main.py \
    --config $work_path/config.yaml --launcher pytorch
