python3 ~/sge/code/onmt_original/OpenNMT-py/embeddings_to_torch.py \
        -emb_file_enc ~/sge/wikipedia/exp3/word-embeddings/embeddings.merged \
        -emb_file_dec ~/sge/wikipedia/exp3/word-embeddings/embeddings.merged \
        -output_file ~/sge/models/exp3_trans/01/corpus/embeddings \
        -dict_file ~/sge/models/exp3_trans/01/corpus/corpus.vocab.pt \
        -type word2vec

