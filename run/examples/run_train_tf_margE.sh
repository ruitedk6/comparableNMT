#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
CUDA_VISIBLE_DEVICES=0

python comparableNMT/train.py \
       	-data ~/path/to/corpus \
       	-save_model ~/path/to/save/tf_margE \
	-layers 6 \
	-transformer_ff 2048 \
	-heads 8 \
	-encoder_type transformer \
	-decoder_type transformer \
	-position_encoding \
	-max_generator_batches 2 \
	-dropout 0.1 \
	-batch_size 50 \
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
	-report_every 50 \
	-share_embeddings \
	-pre_word_vecs_enc ~/path/to/embeddings.enc.pt \
	-pre_word_vecs_dec ~/path/to/embeddings.dec.pt \
	-word_vec_size 512 \
	-rnn_size 512 \
	-comparable \
	-fast \
	-representations embed-only \
	-threshold 1.0 \
	-comp_epochs 10 \
	-threshold_dynamics static \
	-no_base \
	-comparable_data ~/path/to/files.list \
	-comp_log ~/path/to/write/tf_margE \
	-world_size 1 \
	-gpu_ranks 0 \
	&> ~/path/to/write/tf_margE.log
