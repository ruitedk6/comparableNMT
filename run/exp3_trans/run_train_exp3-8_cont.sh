#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
export CUDA_VISIBLE_DEVICES=0

python ~/sge/code/onmt_original/OpenNMT-py/train.py \
       	-data ~/sge/models/exp3_trans/01/corpus/corpus \
       	-save_model ~/sge/models/exp3_trans/08/model/20190301_exp3_8_cont \
	-train_from ~/sge/models/exp3_trans/08/model/20190301_exp3_8_step_0.pt \
	-report_every 50 \
	-share_embeddings \
	-comparable \
	-comp_epochs 10 \
	-threshold_dynamics static \
	-no_base \
	-comparable_data ~/sge/europarl/exp2b/files.list \
	-comp_log ~/sge/models/exp3_trans/08/logs/20190301_exp3_8_cont \
	-world_size 1 \
	-gpu_ranks 0 \
	&> ~/sge/models/exp3_trans/08/logs/20190301_exp3_8_cont.log
