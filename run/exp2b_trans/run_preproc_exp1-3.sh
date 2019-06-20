python3 ~/sge/code/onmt_original/OpenNMT-py/preprocess.py \
	-train_src ~/sge/europarl/exp2b_trans_dummy/train.src \
       	-train_tgt ~/sge/europarl/exp2b_trans_dummy/train.tgt \
       	-valid_src ~/sge/newstest2012/exp1/valid.shuf.src \
       	-valid_tgt ~/sge/newstest2012/exp1/valid.shuf.tgt \
       	-save_data ~/sge/models/exp2b_trans/00/corpus/corpus \
	-comparable \
	-comp_train_src ~/sge/europarl/exp2b_trans_dummy/comp.src \
	-comp_train_tgt ~/sge/europarl/exp2b_trans_dummy/comp.tgt \
	-share_vocab \
	-src_seq_length 80 \
	-tgt_seq_length 80 \
	&> ~/sge/code/onmt_original/OpenNMT-py/logs/20181220_exp2b_trains_00_preproc.log

