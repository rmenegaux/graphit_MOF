#!/bin/bash
#OAR -l walltime=24:00:00
source gpu_setVisibleDevices.sh

GPUID=0

#source /home/rmenegau/.bashrc
export PATH="/scratch/curan/rmenegau/miniconda3/bin:$PATH"
# source activate cwn_tensorboard
source activate matdeeplearn
cd /home/rmenegau/gcn/materials_graphit/transformers
sh zinc_train.sh
