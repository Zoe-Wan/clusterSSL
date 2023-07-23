#!/bin/bash

BASE="/mnt/lustre/suxiu/wzy/swav_main/"
NAME="main_swav"
VERSION="_0"
EPOCH=100
NGPU=8
LR=0.2
ALPHA=0.5
# k-means
K=5000
# dataloader num-worker
WORKERS=6
BATCH=64
T=0.1
EXP="checkpoints/exp_${NAME}${VERSION}/"
CKPT=${EXP}"checkpoint.pth.tar"
PORT=$(( $RANDOM % 300 + 23450 ))
#!/bin/bash

PYTHON="python3"
DATASET="imagenet"
JOB=${BASE}${NAME}".py"


mkdir -p ${EXP}

/mnt/lustre/suxiu/wzy/self-cifar/script/command.sh ${NAME}${VERSION} ${NGPU} 2 "${PYTHON} ${JOB} --port ${PORT} --epochs ${EPOCH}\
 --base_lr ${LR} --feat_dim 256 --hidden_mlp 4096 
--K ${K} --ckpt ${CKPT} --dump_path ${EXP} --batch_size ${BATCH}  --temperature ${T} --alpha ${ALPHA}"


