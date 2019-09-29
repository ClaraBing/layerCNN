#!/bin/bash

sub='1layer'
wandb_name='run1'

CUDA_VISIBLE_DEVICES=2 python tinyImagenet_single_layer.py \
  --ncnn 8 \
  --transform=all \
  --avg_size=8 \
  --save_folder='logs/'$sub \
  --wandb-name=$wandb_name
