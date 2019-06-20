#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
CUDA_VISIBLE_DEVICES=0

python ~/sge/code/onmt_original/OpenNMT-py/train.py \
       	-data ~/sge/models/exp2b/00/corpus/corpus \
       	-save_model ~/sge/models/exp2b/00/model/20181217_exp2b_0 \
	-layers 6 \
	-rnn_size 512 \
	-word_vec_size 512 \
	-transformer_ff 2048 \
	-heads 8 \
	-encoder_type transformer \
	-decoder_type transformer \
	-position_encoding \
	-train_steps 200000 \
	-max_generator_batches 2 \
	-dropout 0.1 \
	-batch_size 4096 \
	-batch_type tokens \
	-normalization tokens \
	-accum_count 2 \
	-optim adam \
	-adam_beta2 0.998 \
	-decay_method noam \
	-warmup_steps 8000 \
	-learning_rate 2 \
	-max_grad_norm 0 \
	-param_init 0 \
	-param_init_glorot \
	-label_smoothing 0.1 \
	-world_size 1 \
	-gpu_ranks 0 \
	&> ~/sge/models/exp2b/00/logs/20181217_exp2b_0.log	
