cd /mnt/DVBE

MODEL=dvbe
DATA=cub
BACKBONE=resnet101
SAVE_PATH=./result_cub

# mkdir -p ${SAVE_PATH}

# nvidia-smi

nohup python main2.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 12 --lr1 0.001 --epochs 180 --resume /mnt/output/fix.model >> /mnt/DVBE/1219_2.log &