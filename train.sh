#!/usr/bin/env bash

nohup python main.py virat Flow   \
   --arch BNInception --num_segments 5 --gpus 0 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 16 -j 8 --dropout 0.7 \
   --snapshot_pref ./models/flow/virat2_bninception \
> nohups/region2/train_flow_bninception_b16_seg5_grad_all.out 2>&1 &