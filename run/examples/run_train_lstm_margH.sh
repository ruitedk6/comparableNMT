#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
CUDA_VISIBLE_DEVICES=0

python comaprableNMT/train.py \
       	-data /path/to/corpus \
       	-save_model /path/to/save/lstm_margH \
	-share_embeddings \
	-pre_word_vecs_enc /path/to/embeddings.enc.pt \
	-pre_word_vecs_dec /path/to/embeddings.dec.pt \
	-encoder_type brnn \
	-layers 1 \
	-global_attention mlp \
	-word_vec_size 512 \
	-rnn_size 512 \
	-learning_rate 0.5 \
	-comp_epochs 10 \
	-comparable \
	-fast \
	-threshold_dynamics static \
	-no_base \
	-comparable_data /path/to/files.list \
	-comp_log /path/to/write/lstm_margH \
	-representations hidden-only \
	-threshold 1.0 \
	-world_size 1 \
	-gpu_ranks 0 \
	&> /path/to/write/lstm_margH.log
