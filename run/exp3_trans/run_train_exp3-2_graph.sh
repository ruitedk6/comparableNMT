#!/bin/bash

#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
#export PATH=${PATH}:/usr/local/cuda/bin
#CUDA_VISIBLE_DEVICES=0

python ~/sge/code/onmt_original/OpenNMT-py/train.py \
       	-data ~/sge/models/exp3_trans/01/corpus/corpus \
       	-save_model ~/sge/models/exp3_trans/00/model/dummy \
	-train_from ~/sge/models/exp3_trans/02/model/20190317_exp3_2_cont_try2_step_2.pt \
	-comparable \
	-comp_epochs 1 \
	-threshold_dynamics static \
	-no_base \
	-fast \
	-write_dual \
	-valid_steps 500 \
	-comparable_data ~/sge/wikipedia/exp5/dummy.list \
	-comp_log ~/sge/models/exp3_trans/00/logs/dummy
