#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
CUDA_VISIBLE_DEVICES=0

python ~/sge/code/onmt_original/OpenNMT-py/train.py \
       	-data ~/sge/models/exp3_trans/01/corpus/corpus \
       	-save_model ~/sge/models/exp3_trans/02/model/20190317_exp3_2_cont_try2 \
	-train_from ~/sge/models/exp3_trans/02/model/20190317_exp3_2_step_149758.pt \
	-learning_rate 0.5 \
	-comparable \
	-comp_epochs 10 \
	-threshold_dynamics static \
	-no_base \
	-fast \
	-valid_steps 500 \
	-comparable_data ~/sge/wikipedia/exp5/files.tc.15.list \
	-comp_log ~/sge/models/exp3_trans/02/logs/20190317_exp3_2_cont_try2 \
	-world_size 1 \
	-gpu_ranks 0 \
	&> ~/sge/models/exp3_trans/02/logs/20190317_exp3_2_cont_try2.log
