#!/usr/bin/env bash

python main.py virat Flow   \
   --arch BNInception --num_segments 3 --gpus 0 1 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 128 -j 8 --dropout 0.7 \
   --snapshot_pref ./models/flow/virat_bninception_flow