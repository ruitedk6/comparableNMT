#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
export CUDA_VISIBLE_DEVICES=0

python ~/sge/code/onmt_original/OpenNMT-py/train.py \
       	-data ~/sge/models/exp3_trans/01/corpus/corpus \
       	-save_model ~/sge/models/exp3_trans/04/model/20190307_exp3_4 \
	-layers 6 \
	-transformer_ff 2048 \
	-heads 8 \
	-encoder_type transformer \
	-decoder_type transformer \
	-position_encoding \
	-train_steps 100 \
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
	-pre_word_vecs_enc ~/sge/models/exp3_trans/01/corpus/embeddings.enc.pt \
	-pre_word_vecs_dec ~/sge/models/exp3_trans/01/corpus/embeddings.dec.pt \
	-word_vec_size 512 \
	-rnn_size 512 \
	-comparable \
	-comp_epochs 10 \
	-representations hidden \
	-threshold 1.01 \
	-threshold_dynamics static \
	-no_base \
	-comparable_data ~/sge/wikipedia/exp5/files.list_below_15 \
	-comp_log ~/sge/models/exp3_trans/04/logs/20190307_exp3_4 \
	-world_size 1 \
	-gpu_ranks 0 \
	&> ~/sge/models/exp3_trans/04/logs/20190307_exp3_4.log
