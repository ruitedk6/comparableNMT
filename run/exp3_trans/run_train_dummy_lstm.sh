#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
CUDA_VISIBLE_DEVICES=0

python ~/sge/code/onmt_original/OpenNMT-py/train.py \
       	-data ~/sge/models/exp3_trans/01/corpus/corpus \
       	-save_model /raid/data/ruiter/models/exp3_trans/00/model/20190213_exp3_2_dummy \
	-dropout 0.1 \
	-batch_size 50 \
	-normalization tokens \
	-report_every 50 \
	-share_embeddings \
	-pre_word_vecs_enc ~/sge/models/exp3_trans/01/corpus/embeddings.enc.pt \
	-pre_word_vecs_dec ~/sge/models/exp3_trans/01/corpus/embeddings.dec.pt \
	-word_vec_size 512 \
	-rnn_size 512 \
	-comparable \
	-comp_epochs 10 \
	-threshold_dynamics static \
	-no_base \
	-world_size 1 \
	-gpu_ranks 0 \
	-comparable_data ~/sge/wikipedia/exp5/files.tc.15.list \
	-comp_log /raid/data/ruiter/models/exp3_trans/00/logs/20190213_exp3_2_dummy
#	&> /raid/data/ruiter/models/exp3_trans/00/logs/20190213_exp3_2_dummy.log
