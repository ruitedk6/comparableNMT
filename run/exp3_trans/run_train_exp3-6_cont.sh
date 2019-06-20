#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
export CUDA_VISIBLE_DEVICES=2

python ~/sge/code/onmt_original/OpenNMT-py/train.py \
       	-data ~/sge/models/exp3_trans/01/corpus/corpus \
       	-save_model ~/sge/models/exp3_trans/06/model/20190223_exp3_6_cont3 \
	-train_from ~/sge/models/exp3_trans/06/model/20190223_exp3_6_cont2_step_17000.pt \
	-report_every 50 \
	-world_size 1 \
	-gpu_ranks 2 \
	-share_embeddings \
	-comparable \
	-comp_epochs 10 \
	-threshold_dynamics static \
	-second \
	-no_base \
	-comparable_data ~/sge/wikipedia/exp5/files.tc.15.list \
	-comp_log ~/sge/models/exp3_trans/06/logs/20190223_exp3_6_cont3 \
	&> ~/sge/models/exp3_trans/06/logs/20190223_exp3_6_cont3.log
