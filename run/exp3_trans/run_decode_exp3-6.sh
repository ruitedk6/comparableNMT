#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=${PATH}:/usr/local/cuda/bin
#export CUDA_VISIBLE_DEVICES=0


while read f; do
	echo $f
	outfile=~/sge/models/exp3_trans/06/eval/${f%.pt}_nt2014_en2fr_pred.txt
	python3 ~/sge/code/onmt_original/OpenNMT-py/translate.py \
		-model ~/sge/models/exp3_trans/06/model/$f \
		-src ~/sge/newstest2014/exp2b/nt.2014.bpe.en \
		-tgt ~/sge/newstest2014/exp2b/nt.2014.bpe.fr \
		-output $outfile \
		-share_vocab \
		-batch_size 1 \
		-gpu 0 \
		-verbose \
		-log_file ~/sge/models/exp3_trans/06/eval/${f%.pt}_nt2014_en2fr_eval.log
	~/sge/code/scripts/prep_eval.sh $outfile fr
done < $1
