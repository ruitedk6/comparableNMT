python3 ~/sge/code/onmt_original/OpenNMT-py/preprocess.py \
	-train_src ~/sge/europarl/exp2_trans_dummy/corpus.src \
       	-train_tgt ~/sge/europarl/exp2_trans_dummy/corpus.tgt \
       	-valid_src ~/sge/newstest2012/exp1/valid.shuf.src \
       	-valid_tgt ~/sge/newstest2012/exp1/valid.shuf.tgt \
       	-save_data ~/sge/models/exp2b_trans/00/corpus/corpus_no_comp \
	-share_vocab \
	-src_seq_length 80 \
	-tgt_seq_length 80 \
	&> ../../logs/20181217_exp1_0_preproc.log

