python3 ~/sge/code/onmt_original/OpenNMT-py/preprocess.py \
	-train_src ~/sge/wikipedia/exp5/wp.cat.src \
	-train_tgt ~/sge/wikipedia/exp5/wp.cat.tgt \
       	-valid_src ~/sge/newstest2012/exp2b/valid.shuf.src \
       	-valid_tgt ~/sge/newstest2012/exp2b/valid.shuf.tgt \
       	-save_data ~/sge/models/exp3_trans/01/corpus/corpus \
	-src_vocab ~/sge/wikipedia/exp2b/vocabulary.src \
	-tgt_vocab ~/sge/wikipedia/exp2b/vocabulary.tgt \
	-src_vocab_size 93210 \
	-tgt_vocab_size 93210 \
	-share_vocab \
	-shard_size 5000000 \
	-src_seq_length 50 \
	-tgt_seq_length 50 \
	&> ~/sge/code/onmt_original/OpenNMT-py/logs/20190128_exp3_trans_01_preproc.log

