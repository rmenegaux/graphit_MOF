#!/bin/bash


############
# Usage
############

# bash script_ZINC_all.sh


####################################
# ZINC - 4 SEED RUNS OF EACH EXPTS
####################################

seed0=41
seed1=95
seed2=12
seed3=35
code=main_zinc_regression.py 
dataset=ZINC
config='GraphiT_ZINC_NoPE.json'
#config='GraphiT_ZINC_gckn_marg.json'
# tmux new -s gnn_lspe_ZINC -d
# tmux send-keys "source ~/.bashrc" C-m
# tmux send-keys "source activate gnn_lspe" C-m
# tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config $config #&
# python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'GraphiT_ZINC_NoPE.json' &
# python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'GraphiT_ZINC_NoPE.json' &
# python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'GraphiT_ZINC_NoPE.json' #&
# wait" C-m
# tmux send-keys "tmux kill-session -t gnn_lspe_ZINC" C-m












