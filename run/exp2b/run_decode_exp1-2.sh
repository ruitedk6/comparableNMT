#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
CUDA_VISIBLE_DEVICES=3


python3 ~/sge/code/onmt_original/OpenNMT-py/translate.py \
	-model ~/sge/models/exp2b/00/model/20181217_exp2b_0_step_15000.pt \
	-src ~/sge/models/exp2b/00/eval/test2.src \
	-tgt ~/sge/models/exp2b/00/eval/test2.tgt \
	-output ~/sge/models/exp2b/00/eval/pred.txt \
	-report_bleu \
	-share_vocab \
	-verbose \
	-log_file ~/sge/models/exp2b/00/eval/20181218_exp2b_0_eval.log \
	-gpu 3
