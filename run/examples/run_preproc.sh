python3 comparableNMT/preprocess.py \
	-train_src /path/to/corpus.src \
	-train_tgt /path/to/corpus.tgt \
       	-valid_src /path/to/valid.src \
       	-valid_tgt /path/to/valid.tgt \
       	-save_data /path/to/save/corpus \
	-src_vocab /path/to/vocabulary.src \
	-tgt_vocab /path/to/vocabulary.tgt \
	-src_vocab_size 93210 \
	-tgt_vocab_size 93210 \
	-share_vocab \
	-shard_size 5000000 \
	-src_seq_length 50 \
	-tgt_seq_length 50 \
	&> ~/path/to/write/preproc.log

