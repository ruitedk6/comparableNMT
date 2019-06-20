#!/bin/bash


python ~/sge/code/onmt_original/OpenNMT-py/train.py \
       	-data ~/sge/models/exp2b_trans/00/corpus/corpus \
       	-save_model ~/sge/models/exp_trans/00/model/dummy \
	-layers 6 \
	-rnn_size 512 \
	-word_vec_size 512 \
	-transformer_ff 2048 \
	-heads 8 \
	-encoder_type transformer \
	-decoder_type transformer \
	-position_encoding \
	-train_steps 4 \
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
	-report_every 1 \
	-comparable \
	-comp_epochs 2 \
	-threshold_dynamics grow \
	-infer_threshold percentile \
	-infer_data base \
	-comp_example_limit 80 \
	-comp_log ~/sge/models/exp2b_trans/00/logs/dummy
