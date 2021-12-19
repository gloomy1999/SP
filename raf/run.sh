#!/bin/bash

cd /mnt/DVBE

MODEL=dvbe
DATA=raf
BACKBONE=resnet101
SAVE_PATH=/mnt/output

# mkdir -p ${SAVE_PATH}

# nvidia-smi

nohup python main2.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 128 --pretrained --is_fix >> /mnt/DVBE/fix_log/1219.log &

# python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 12 --lr1 0.001 --epochs 180 --resume ${SAVE_PATH}/fix.model &> ${SAVE_PATH}/ft.txt
