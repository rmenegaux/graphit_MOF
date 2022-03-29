#!/bin/bash
#OAR -l walltime=48:00:00
#OAR -O out_oar/%jobid%.stdout
#OAR -E out_oar/%jobid%.stderr

source gpu_setVisibleDevices.sh

GPUID=0

#export PATH="/scratch/prospero/mselosse/miniconda3/bin:$PATH"
conda activate tb
cd /scratch/prospero/mselosse/graph_transformer
sh zinc_train.sh