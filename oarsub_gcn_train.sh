#!/bin/bash
#OAR -l walltime=24:00:00
source gpu_setVisibleDevices.sh

GPUID=0

#source /home/rmenegau/.bashrc
export PATH="/scratch/curan/rmenegau/miniconda3/bin:$PATH"
source activate cwn_tensorboard
cd /home/rmenegau/gcn/transformers
sh zinc_train.sh