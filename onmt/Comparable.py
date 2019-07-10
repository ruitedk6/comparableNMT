"""
Classes and methods used for training and extraction of parallel pairs
from a comparable dataset.
Author: Dana Ruiter
"""
import torch
import torch.nn as nn
from torchtext.data import Batch

import onmt
import onmt.utils
from onmt.trainer import Trainer
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset
import onmt.inputters as inputters

import math
from collections import defaultdict
import numpy as np
import time

class CompExample():
    """
    Class that stores the information of one parallel data example.
    Args:
        dataset(:obj:'onmt.io.TextDataset.TextDataset'): dataset object
        fields(list): list of keys of fields in dataset object
        src(torch.Tensor): src sequence (size(seq))
        tgt(torch.Tensor): tgt sequence (size(seq))
        src_length(torch.Tensor): the length of the src sequence (size([]))
        index(torch.Tensor): the index of the example in dataset
    """
    # These should be the same for all examples (else: consistency problem)
    _dataset = None
    _fields = None

    def __init__(self, dataset, fields, src, tgt, src_length, index):
        self.src = src
        self.tgt = tgt
        self.src_length = src_length
        self.index = index


        if CompExample._dataset == None:
            CompExample._dataset = dataset

        if CompExample._fields == None:
            CompExample._fields = fields

    def get_dataset():
        return CompExample._dataset

    def get_fields():
        return CompExample._fields

class PairBank():
    """
    Class that saves and prepares parallel pairs and their resulting
    batches.

    Args:
        batch_size(int): number of examples in a batch
        opt(argparse.Namespace): option object
    """
    def __init__(self, batch_size, opt):
        self.pairs = []
        self.index_memory = set()
        self.batch_size = batch_size
        self.limit = opt.comp_example_limit
        self.use_gpu = (len(opt.gpu_ranks) > 0)

    def removePadding(side):
        """ Removes original padding from a sequence.
        Args:
            side(torch.Tensor): src/tgt sequence (size(seq))

        Returns:
            side(torch.Tensor): src/tgt sequence without padding
        NOTE: This only works as long as PAD_ID==1!
        """
        # Get indexes of paddings in sequence
        padding_idx = (side == 1).nonzero()
        # If there is any padding, cut sequence from first occurence of a pad
        if padding_idx.size(0) != 0:
            first_pad = padding_idx.data.tolist()[0][0]
            side = side[:first_pad]
        return side


    def add_example(self, src, tgt, fields):
        """ Add an example from a batch to the PairBank (self.pairs).
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
            fields(list(str)): list of keys of fields
        """
        # Get example from src/tgt and remove original padding
        src = PairBank.removePadding(src)
        tgt = PairBank.removePadding(tgt)
        if self.use_gpu:
            src_length = torch.tensor([src.size(0)]).cuda()
        else:
            src_length = torch.tensor([src.size(0)])
        index = None
        # Create CompExample object holding all information needed for later
        # batch creation.
        example = CompExample(None, fields, src, tgt,
                              src_length, index)
        # Add to pairs
        self.pairs.append(example)
        # Remember unique src-tgt combination
        self.index_memory.add(hash((str(src), str(tgt))))
        return None

    def contains_batch(self):
        """Check if enough parallel pairs found to create a batch.
        """
        return (len(self.pairs) >= self.batch_size)

    def no_limit_reached(self, src, tgt):
        """ Check if no assigned limit of unique src-tgt pairs is reached.
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
        """
        src = PairBank.removePadding(src)
        tgt = PairBank.removePadding(tgt)
        return (hash((str(src), str(tgt))) in self.index_memory or len(self.index_memory) < self.limit)

    def get_max_length_sequence(examples):
        """Get the length of the longes sequence in list of examples"""
        return max([ex.size(0) for ex in examples])

    def shape_example(self, example, max_len):
        """ Formats an example to fit with its batch.
        Args:
            example(torch.Tensor): a src/tgt sequence (size(seq))
        """
        # Add batch dimension
        example = example.unsqueeze(1)
        # Pad to max_len if necessary
        pad_size = max_len - example.size(0)
        if pad_size != 0:
            if self.use_gpu:
                pad = torch.ones(pad_size, 1, dtype=torch.long).cuda()
            else:
                pad = torch.ones(pad_size, 1, dtype=torch.long)
            example = torch.cat((example, pad), 0)
        return example

    def preprocess_side(self, examples):
        """ Formats a list of examples into one tensor.
        Args:
            examples(list): list of src/tgt sequence tensors
        Returns:
            batch(torch.Tensor): src/tgt side of the batch (size(seq, batch))
        """
        max_len = PairBank.get_max_length_sequence(examples)
        examples = [self.shape_example(ex, max_len) for ex in examples]
        # Concatenate examples along the batch axis
        batch = torch.cat(examples, 1)
        return batch

    def sort(src_examples, tgt_examples, src_lengths, indices):
        """ Sort examples based on descending src_lengths.
        Args:
            src_examples(list): list of src sequence tensors
            tgt_examples(list): list of tgt sequence tensors
            src_lengths(list): list of the lengths of each src sequence
            indices(list): list of indices of example instances in dataset
        """
        examples = zip(src_examples, tgt_examples, src_lengths, indices)
        examples = sorted(examples, key=lambda x: x[2].item(), reverse=True)
        return zip(*examples)
    
    def check_sos_eos(self, tgt_example):
        # If there is no sos symbol, add it
        if tgt_example[tgt_example==2].shape[0] == 0:
            if self.use_gpu:
                sos = torch.tensor([2]).cuda()
            else:
                sos = torch.tensor([2])
            tgt_example = torch.cat((sos, tgt_example), 0)
        # If there is no eos symbol, add it
        if tgt_example[tgt_example==3].shape[0] == 0:
            if self.use_gpu:
                eos = torch.tensor([3]).cuda()
            else:
                
                eos = torch.tensor([3])
            tgt_example = torch.cat((tgt_example, eos), 0)
        return tgt_example

    def create_batch(self, src_examples, tgt_examples, src_lengths, indices, dataset, fields):
        """ Creates a batch object from previously extracted parallel data.
        Args:
            src_examples(list): list of src sequence tensors
            tgt_examples(list): list of tgt sequence tensors
            src_lenths(list): list of the lengths of each src sequence
            indices(list): list of indices of example instances in dataset
            dataset(:obj:'onmt.io.TextDataset.TextDataset'): dataset object
            fields(list): list of keys of fields in dataset object

        Returns:
            batch(torchtext.data.batch.Batch): batch object
        """
        batch = Batch()
        src_examples, tgt_examples, src_lengths, indices = \
            PairBank.sort(src_examples, tgt_examples, src_lengths, indices)
        src = self.preprocess_side(src_examples)
        tgt_examples = [self.check_sos_eos(ex) for ex in tgt_examples]
        tgt = self.preprocess_side(tgt_examples)
        src_lengths = torch.cat([length for length in src_lengths])
        indices = None

        batch.batch_size = src.size(1)
        batch.dataset = dataset
        batch.fields = fields
        batch.train = True
        batch.src = (src, src_lengths)
        batch.tgt = tgt
        batch.indices = indices

        return batch

    def get_num_examples(self):
        """Returns batch size if no maximum number of extracted parallel data
        used for training is met. Otherwise returns number of examples that can be yielded
        without exceeding that maximum.
        """
        if len(self.pairs) < self.batch_size:
            return len(self.pairs)
        return self.batch_size

    def yield_batch(self):
        """ Prepare and yield a new batch from self.pairs.

        Returns:
            batch(torchtext.data.batch.Batch): batch of extracted parallel data
        """
        src_examples = []
        tgt_examples = []
        src_lengths = []
        indices = []
        num_examples = self.get_num_examples()

        # Get as many examples as needed to fill a batch or a given limit
        for ex in range(num_examples):
            example = self.pairs.pop()
            src_examples.append(example.src)
            tgt_examples.append(example.tgt)
            src_lengths.append(example.src_length)
            indices.append(example.index)

        dataset = None
        fields = CompExample.get_fields()
        batch = self.create_batch(src_examples, tgt_examples, src_lengths, indices,
                             dataset, fields)
        return batch

class CompTrainer(Trainer):
    """
    Class that manages the training on extracted parallel data.

    Args:
        trainer(:obj:'onmt.Trainer.Trainer'):
            trainer object used for training on parallel data
        logger(logging.RootLogger): logger that reports about training
        report_func(fn): function for logging
        opt(argparse.Namespace): option object
    """
    def __init__(self, trainer, logger, opt):
        super().__init__(trainer.model, trainer.train_loss, trainer.valid_loss,
              trainer.optim, trainer.trunc_size, trainer.shard_size,
              trainer.data_type, trainer.norm_method, trainer.grad_accum_count, trainer.n_gpu,
                         trainer.gpu_rank, trainer.gpu_verbose_level, trainer.report_manager,
                         trainer.model_saver)
        self.cur_epoch = 0
        self.logger = logger
        self.opt = opt

    def next_epoch(self):
        """Initialize new epoch."""
        self.true_batchs = []
        self.accum = 0
        self.normalization = 0
        self.total_stats = onmt.utils.Statistics()
        self.report_stats = onmt.utils.Statistics()
        self.cur_epoch += 1
        self.idx = 0

    def report_func(self, epoch, batch, num_batches, start_time, lr, report_stats):
        """ Original report function."""
        if batch % self.opt.report_every == -1 % self.opt.report_every:
            t = report_stats.elapsed_time()
            msg = (("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; xent: " +
                    "%6.2f; lr: %7.5f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s " + 
                   "elapsed") %
                  (epoch, batch + 1, num_batches, 
                   report_stats.accuracy(), 
                   report_stats.ppl(),
                   report_stats.xent(), 
                   lr,
                   report_stats.n_src_words / (t + 1e-5),
                   report_stats.n_words / 
                   (t + 1e5), time.time() - start_time))
            self.logger.info(msg)
            report_stats = onmt.utils.Statistics()
        return report_stats

    def train(self, batch):
        """ Train model on one batch.
        Args:
            batch(torchtext.data.batch.Batch): batch to be trained on
        
        Returns:
            self.total_stats(:obj:'onmt.Statistics'): loss statistics
        """

        self.true_batchs.append(batch)
        if self.norm_method == "tokens":
            num_tokens = batch.tgt[1:].ne(
                self.train_loss.padding_idx).sum()
            self.normalization += num_tokens.item()
        else:
            self.normalization += batch.batch_size
        self.accum += 1
        if self.accum == self.grad_accum_count:
            # F-prop

            self._gradient_accumulation(self.true_batchs, self.normalization,
                                    self.total_stats, self.report_stats)
            # Report
            self.report_stats = self.report_func(self.cur_epoch, self.idx, -1,
                                        self.total_stats.start_time, self.optim.learning_rate,
                                        self.report_stats)
            self.true_batchs = []
            self.accum = 0
            self.normalization = 0
        # Prepare for next batch
        self.idx += 1
        return self.total_stats

    def report_epoch(self):
        """ Report statistics of current epoch."""
        self.logger.info('Train perplexity: %g' % self.total_stats.ppl())
        self.logger.info('Train accuracy: %g' % self.total_stats.accuracy())
        return None

    def validate(self, valid_iter):
        """ Validate on validation data.
        Args:
            valid_iter(:obj:'train.DatasetLazyIter'): validation data iterator

        Returns:
            valid_stats(:obj:'onmt.Statistics'): validation loss statistics
        """
        valid_stats = super(CompTrainer, self).validate(valid_iter)
        self.logger.info('Validation perplexity: %g' % valid_stats.ppl())
        self.logger.info('Validation accuracy: %g' % valid_stats.accuracy())
        return valid_stats

class Comparable():
    """
    Class that controls the extraction of parallel sentences and manages their
    storage and training.

    Args:
        model(:py:class:'onmt.Model.NMTModel'):
            translation model used for extraction and training
        trainer(:obj:'onmt.Trainer.Trainer'): 
            trainer that controlls the training process
        fields(dict): fields and vocabulary
        logger(logging.RootLogger):
            logger that reports information about extraction and training
        opt(argparse.Namespace): option object
    """

    def __init__(self, model, trainer, fields, logger, opt):
        self.sim_measure = opt.sim_measure
        self.threshold = opt.threshold
        self.similar_pairs = PairBank(opt.batch_size, opt)
        self.trainer = CompTrainer(trainer, logger, opt)
        self.encoder = self.trainer.model.encoder
        self.fields = fields
        self.logger = logger
        self.accepted = 0
        self.accepted_limit = 0
        self.declined = 0
        self.total = 0
        self.comp_log = opt.comp_log
        self.cove_type = opt.cove_type
        self.k = 4
        self.opt = opt
        self.gpu = torch.device('cuda') if len(opt.gpu_ranks) > 0 else None
        self.trainstep = 0
        self.second = opt.second
        self.representations = opt.representations
        self.max_len = opt.max_len
        self.valid_steps = opt.valid_steps
        self.no_valid = opt.no_valid
        self.fast = opt.fast
        self.write_dual = opt.write_dual
        self.no_swaps = opt.no_swaps


    def _get_iterator(self, path):
        """
        Creates an iterator object from a text file.

        Args:
            path(str): path to text file to process
        Returns:
            data_iter(inputters.OrderedIterator): iterator object
        """
        # Create dataset object
        data = inputters.build_dataset(fields=self.fields,
                                     data_type='text',
                                     src_path=path,
                                     tgt_path=None,
                                     src_dir='',
                                     use_filter_pred=False)

        data_iter = inputters.OrderedIterator(dataset=data,
                                            device=self.gpu,
                                            batch_size=self.similar_pairs.batch_size,
                                            train=False,
                                            sort=False,
                                            sort_within_batch=True,
                                            shuffle=False)
        return data_iter

    def replaceEOS(fets):
        """Gets rid of EOS and SOS symbols.
        Args:
            fets(torch.Tensor): src/tgt tensor (size(seq, batch, 1))
        Note: This only works when EOS==3!
        Returns:
            fets(torch.Tensor): src/tgt tensor (size(seq, batch, 1))
        """
        fets[fets==3] = 1
        return fets[:-1 , :, :]

    def getFeatures(batch, side):
        """ Gets the tensor of a src/tgt side from the batch.
        Args:
            batch(torchtext.data.batch.Batch): batch object
            side(str): name of the side

        Returns:
            fets(torch.Tensor): tensor of a side (size(seq, batch, 1))
            lengths(torch.Tensor): lengths of each src sequence (None if tgt)
        """
        fets = onmt.inputters.make_features(batch, side, 'text')
        lengths = None
        if side == 'tgt':
            fets = fets[1: , :, :]
            # Remove EOS and SOS symbols to make comparable with src sequences
            fets = Comparable.replaceEOS(fets)
        if side == 'src':
            _, lengths = batch.src
        return fets, lengths

    def forward(self, side, representation='memory'):
        """ F-prop a src or tgt batch through the encoder.
        Args:
            side(torch.Tensor): batch to be f-propagated
                (size(seq, batch, 1))
            representation(str): if 'memory', access hidden states; else embeddings

        Returns:
            memory_bank(torch.Tensor): encoder outputs
                (size(seq, batch, fets))
        """
        # Do not accumulate gradients
        with torch.no_grad():
            if representation == 'embed':
                # word embeddings
                embeddings = self.encoder.embeddings(side)
                return embeddings
            else: 
                # hidden states/encoder output
                embeddings, memory_bank, src_lengths = self.encoder(side, None)
                return memory_bank

    def calculate_similarity(self, src, tgt):
        """ Calculates the cosine similarity between two sentence representations.
        Args:
            src(torch.Tensor): src sentence representation (size(fets))
            tgt(torch.Tensor): tgt sentence representation (size(fets))
        Returns:
            float: cosine similarity
        """
        return nn.functional.cosine_similarity(src, tgt, dim=0).tolist()

    def idx2words(self, seq, side):
        """ Convert word indices to words.
        Args:
            seq(torch.tensor): a src/tgt sequence (size(seq))
            side(str): {'src'|'tgt'}
        Returns:
            words(list(str)): list of words in the sequence
        """
        vocab = self.fields[side].vocab.itos
        words = [vocab[idx] for idx in seq.data.tolist() \
                 if idx not in [0, 1, 2, 3]]
        return words

    def write_sentence(self, src, tgt, status, score=None):
        """
        Writes an accepted parallel sentence candidate pair to a file.

        Args:
            src(torch.tensor): src sequence (size(seq))
            tgt(torch.tensor): tgt sequence (size(seq))
            status(str): ['accepted', 'accepted-limit', 'rejected']
            score(float): score of the sentence pair
        """ 
        src_words = self.idx2words(src, 'src')
        tgt_words = self.idx2words(tgt, 'tgt')
        out = 'src: {}\ttgt: {}\tsimilarity: {}\tstatus: {}\n'.format(' '.join(src_words), 
                                            ' '.join(tgt_words), score, status)
        if status == 'accepted' or status == 'accepted-limit':
            self.accepted_file.write(out)
        elif status == 'embed_only':
            with open(self.embed_file, 'a', encoding='utf8') as f:
                f.write(out)
        elif status == 'hidden_only':
            with open(self.hidden_file, 'a', encoding='utf8') as f:
                f.write(out)
        return None

    def get_cove(self, memory, ex, mean=False):
        """ Get sentence representation.
        Args:
            memory(torch.Tensor): hidden states or word embeddings of batch
            ex(int): index of example in batch
            mean(boolean): if true, take mean over time-steps; else, sum
        Returns:
            cove(torch.Tensor): sentence representation C_e or C_h
        """
        # Get current example
        seq_ex = memory[:, ex, :]
        if mean:
            cove = torch.mean(seq_ex, dim=0)
        else:
            cove = torch.sum(seq_ex, dim=0)
        return cove


    def extract_parallel_sents(self, candidates, candidate_pool):
        """ 
        Extracts parallel sentences from candidates and adds them to the
        PairBank (secondary filter).

        Args:
            candidates(list(tuple(torch.Tensor...)): list of src-tgt candidates
            candidate_pool(list(hash)): list of hashed C_e candidates
        """
        for candidate in candidates:
            candidate_pair = hash((str(candidate[0]), str(candidate[1])))
            # For dual representation systems...
            if candidate_pool:
                # ...skip C_h pairs not in C_e (secondary filter)
                if candidate_pair not in candidate_pool:
                    self.declined += 1
                    self.total += 1
                    if self.write_dual:
                        self.write_sentence(candidate[0], candidate[1],
                                            'hidden_only', candidate[2])
                    continue
            if self.no_swaps:
                swap = False
            # Swap src-tgt direction randomly
            else:
                swap = np.random.randint(2)
            if swap:
                src = candidate[1]
                tgt = candidate[0]
            else:
                src = candidate[0]
                tgt = candidate[1]

            score = candidate[2]
            # Apply threshold (single-representation systems only)
            if score >= self.threshold:
                # Check if no maximum of allowed unique accepted pairs reached
                if self.similar_pairs.no_limit_reached(src, tgt):
                    # Add to PairBank
                    self.similar_pairs.add_example(src, tgt, self.fields)
                    self.accepted += 1
                    self.write_sentence(src, tgt, 'accepted', score)
                else:
                    self.accepted_limit += 1
                    self.write_sentence(src, tgt, 'accepted-limit', score)

            else:
                self.declined +=1
            self.total += 1

        return None
 
    def write_similarities(self, values, name):
        """ Write epoch scores to file.
        Args:
            values(list(float)): list of scores
            name(str): epoch name
        """
        val_count = defaultdict(int)
        for val in values:
            val_count[round(val, 2)] += 1

        with open("{}_{}_similarities.tsv".format(self.comp_log, name), 'w+', encoding='utf8') as out:
            for val in list(val_count.keys()):
                out.write("{}\t{}\n".format(val, val_count[val]))



    def _sum_k_nearest(self, mapping, cove):
        """ Calculates average score of a sequence to its k-nearest neighbors.
        Args:
            mapping(dict(dict(float))): L1-L2 mapping with their respective cosine sim
            cove(torch.Tensor): sentence representation of L1 sequence
        Returns:
            float: denominator of margin-based scoring function
        """
        # Get k-nearest neighbors
        k_nearest = sorted(mapping[cove].items(), key=lambda x: x[1], reverse=True)[:self.k]
        # Sum scores and return denominator
        sum_k_nearest = sum([ex[1] for ex in k_nearest])
        return sum_k_nearest / (2 * len(k_nearest))

    def score_sents(self, src_sents, tgt_sents):
        """ Score source and target combinations.
        Args:
            src_sents(list(tuple(torch.Tensor...))):
                list of src sentences in their sequential and semantic representation
            tgt_sents(list(tuple(torch.Tensor...))): list of tgt sentences
        Returns:
            src2tgt(dict(dict(float))): dictionary mapping a src to a tgt and their score
            tgt2src(dict(dict(float))): dictionary mapping a tgt to a src and their score
            similarities(list(float)): list of cosine similarities
            scores(list(float)): list of scores
        """
        src2tgt = defaultdict(dict)
        tgt2src = defaultdict(dict)
        similarities = []
        scores= []
        for src, src_cove in src_sents:
            for tgt, tgt_cove in tgt_sents:
                # Ignore combination if both sentences from the same language
                if src[0] == tgt[0]:
                    continue
                # Calculate cosine similarity of combination
                sim = self.calculate_similarity(src_cove, tgt_cove)
                src2tgt[src][tgt] = sim
                tgt2src[tgt][src] = sim
                similarities.append(sim)
        # Return cosine similarity if that is the scoring function
        if self.sim_measure == 'cosine':
            return src2tgt, tgt2src, similarities, similarities

        # Else, continue to calculate margin-based score
        # Calculate denominator (average cosine similarity to k-nearest neighbors)
        for src, _ in src_sents:
            src2tgt[src]['sum'] = self._sum_k_nearest(src2tgt, src)

        for tgt, _ in tgt_sents:
            tgt2src[tgt]['sum'] = self._sum_k_nearest(tgt2src, tgt)

        for src, _ in src_sents:
            for tgt, _ in tgt_sents:
                if src[0] == tgt[0]:
                    continue
                # Apply denominator to each combination...
                src2tgt[src][tgt] /= (src2tgt[src]['sum'] + tgt2src[tgt]['sum'])

        for tgt, tgt_cove in tgt_sents:
            for src, src_cove in src_sents:
                if src[0] == tgt[0]:
                    continue
                # ... in both language directions
                tgt2src[tgt][src] /= (src2tgt[src]['sum'] + tgt2src[tgt]['sum'])
            del tgt2src[tgt]['sum']

        # Get list of scores for statistics
        for src in list(src2tgt.keys()):
            del src2tgt[src]['sum']
            scores += list(src2tgt[src].values())

        return src2tgt, tgt2src, similarities, scores


    def get_article_coves(self, article, representation='memory', fast=False, mean=False):
        """ Get representations (C_e or C_h) for sentences in a document.
        Args:
            article(inputters.OrderedIterator): iterator over sentences in document
            representation(str): if 'memory', create C_h; if 'embed', create C_e
            fast(boolean): if true, only look at first batch in article
            mean(boolean): if true, use mean over time-step representations; else, sum
        Returns:
            sents(list(tuple(torch.Tensor...))):
                list of sentences in their sequential (seq) and semantic representation (cove)
        """
        sents = []
        for batch in article:
            # Decompress
            fets, _ = Comparable.getFeatures(batch, 'src')

            if representation == 'memory':
                # Get hidden states
                sent_repr = self.forward(fets)
            elif representation == 'embed':
                # Get word-embeddings
                sent_repr = self.forward(fets, representation='embed')

            for ex in range(fets.size(1)):
                # Compress to sentence representations C_e or C_h
                cove = self.get_cove(sent_repr, ex, mean=mean)
                seq = batch.src[0][:, ex]
                # Remove long sequences
                if batch.src[1][ex].item() > self.max_len:
                    continue
                # Remove empty sequences
                if seq[0].item() == 1:
                    continue
                sents.append((seq, cove))

            if fast:
                return sents

        return sents

    def filter_candidates(self, src2tgt, tgt2src, second=False):
        """ Filter candidates (primary filter), such that only those which are top candidates in
        both src2tgt and tgt2src direction pass.
        Args:
            src2tgt(dict(dict(float))): mapping src sequence to tgt sequence and score
            tgt2src(dict(dict(float))): mapping tgt sequence to src sequence and score
            second(boolean): if true, also include second-best candidate for src2tgt direction
                (medium permissibility mode only)
        Returns:
            candidates(list(tuple(torch.Tensor...)): list of src-tgt candidates
        """
        src_tgt_max = set()
        tgt_src_max = set()
        src_tgt_second = set()
        tgt_src_second = set()

        # For each src...
        for src in list(src2tgt.keys()):
            toplist = sorted(src2tgt[src].items(), key=lambda x: x[1], reverse=True)
            # ... get the top scoring tgt
            max_tgt = toplist[0]
            # Get src, tgt and score
            src_tgt_max.add((src, max_tgt[0], max_tgt[1]))
            if second:
                # If high permissibility mode, also get second-best tgt
                second_tgt = toplist[1]
                src_tgt_second.add((src, second_tgt[0], second_tgt[1]))

        # For each tgt...
        for tgt in list(tgt2src.keys()):
            toplist = sorted(tgt2src[tgt].items(), key=lambda x: x[1], reverse=True)
            # ... get the top scoring src
            max_src = toplist[0]
            tgt_src_max.add((max_src[0], tgt, max_src[1]))

        if second:
            # Intersection as defined in medium permissibility mode
            src_tgt = (src_tgt_max | src_tgt_second) & tgt_src_max
            candidates = list(src_tgt)
            return candidates

        # Intersection as defined in low permissibility
        candidates = list(src_tgt_max & tgt_src_max)
        return candidates

    def get_comparison_pool(self, src_embeds, tgt_embeds):
        """ Perform scoring and filtering for C_e (in dual representation system)
        Args:
            src_embeds(list): list of source embeddings (C_e)
            tgt_embeds(list): list of target embeddings (C_e)
        Returns:
            candidate_pool(set): set of hashed src-tgt C_e pairs
            candidate_embed(list): list of src-tgt C_e pairs
        """
        # Scoring
        src2tgt_embed, tgt2src_embed, _, _ = self.score_sents(src_embeds, tgt_embeds)
        # Filtering (primary filter)
        candidates_embed = self.filter_candidates(src2tgt_embed, tgt2src_embed)
        # Create set of hashed pairs (for easy comparison in secondary filter)
        set_embed = set([hash((str(c[0]), str(c[1]))) for c in candidates_embed])
        candidate_pool = set_embed
        return candidate_pool, candidates_embed


    def write_embed_only(self, candidates, cand_embed):
        """ Writes C_e scores to file (if --write-dual is set).
        Args:
            candidates(list): list of src, tgt pairs (C_h)
            cand_embed(list): list of src, tgt pairs (C_e)
        """
        candidate_pool = set([hash((str(c[0]), str(c[1]))) for c in candidates])

        for candidate in cand_embed:
            candidate_pair = hash((str(candidate[0]), str(candidate[1])))
            # Write statistics only if C_e pair not in C_h
            if candidate_pair not in candidate_pool:
                src = candidate[0]
                tgt = candidate[1]
                score = candidate[2]
                self.write_sentence(src, tgt, 'embed_only', score)

    def extract_and_train(self, comparable_data_list):
        """ Manages the alternating extraction of parallel sentences and training.
        Args:
            comparable_data_list(str): path to list of mapped documents
        Returns:
            train_stats(:obj:'onmt.Trainer.Statistics'): epoch loss statistics
        """
        # Start first epoch
        self.trainer.next_epoch()
        self.accepted_file = \
                open('{}_accepted-e{}.txt'.format(self.comp_log, self.trainer.cur_epoch), 'w+', encoding='utf8')
        self.status_file = '{}_status-e{}.txt'.format(self.comp_log, self.trainer.cur_epoch)
        if self.write_dual:
            self.embed_file = '{}_accepted_embed-e{}.txt'.format(self.comp_log,
                                                                 self.trainer.cur_epoch)
            self.hidden_file = '{}_accepted_hidden-e{}.txt'.format(self.comp_log,
                                                                 self.trainer.cur_epoch)
        epoch_similarities = []
        epoch_scores = []
        counter = 0
        src_sents = []
        tgt_sents = []
        src_embeds = []
        tgt_embeds = []

        # Go through comparable data
        with open(comparable_data_list, encoding='utf8') as c:
            comp_list = c.read().split('\n')
            num_articles = len(comp_list)
            cur_article = 0

            for article_pair in comp_list:
                cur_article += 1

                # Update status
                with open(self.status_file, 'a', encoding='utf8') as sf:
                    sf.write('{} / {}\n'.format(cur_article, num_articles))

                articles = article_pair.split('\t')
                # Discard malaligned documents
                if len(articles) != 2:
                    continue

                # Prepare iterator objects for current src/tgt document
                src_article = self._get_iterator(articles[0])
                tgt_article = self._get_iterator(articles[1])

                # Get sentence representations
                try:
                    if self.representations == 'embed-only':
                        # C_e
                        src_sents += self.get_article_coves(src_article, 'embed', fast=self.fast)
                        tgt_sents += self.get_article_coves(tgt_article, 'embed', fast=self.fast)
                    else:
                        # C_h
                        src_sents += self.get_article_coves(src_article, fast=self.fast)
                        tgt_sents += self.get_article_coves(tgt_article, fast=self.fast)
                        # C_e
                        src_embeds += self.get_article_coves(src_article,
                                                            'embed', fast=self.fast)
                        tgt_embeds += self.get_article_coves(tgt_article,
                                                        'embed', fast=self.fast)
                except:
                    # Skip document pair in case of errors
                    src_sents = []
                    tgt_sents = []
                    src_embeds = []
                    tgt_embeds = []
                    continue

                # Ensure enough sentences are accumulated (otherwise scoring becomes unstable)
                if len(src_sents) < 15 or len(tgt_sents) < 15:
                    continue

                # Score src and tgt sentences
                src2tgt, tgt2src, similarities, scores = self.score_sents(src_sents, tgt_sents)
                
                # Keep statistics
                epoch_similarities += similarities
                epoch_scores += scores
                src_sents = []
                tgt_sents = []
                
                # Filter candidates (primary filter)
                try:
                    if self.representations == 'dual':
                        # For dual representation systems, filter C_h...
                        candidates = self.filter_candidates(src2tgt, tgt2src, second=self.second)
                        # ...and C_e
                        comparison_pool, cand_embed = self.get_comparison_pool(src_embeds,
                                                                               tgt_embeds)
                        src_embeds = []
                        tgt_embeds = []
                        if self.write_dual:
                            self.write_embed_only(candidates, cand_embed)
                    else:
                        # Filter C_e or C_h for single representation system
                        candidates = self.filter_candidates(src2tgt, tgt2src)
                        comparison_pool = None
                except:
                    # Skip document pair in case of errors
                    print('Error occured in: {}\n'.format(article_pair), flush=True)
                    src_embeds = []
                    tgt_embeds = []
                    continue

                # Extract parallel samples (secondary filter)
                self.extract_parallel_sents(candidates, comparison_pool)
                
                # Check if enough parallel sentences were collected
                while self.similar_pairs.contains_batch():
                    # Get a batch of extracted parrallel sentences and train
                    try:
                        training_batch = self.similar_pairs.yield_batch()
                    except:
                        print('Error creating batch. Continuing...', flush=True)
                        continue

                    # Statistics
                    train_stats = self.trainer.train(training_batch)
                    self.trainstep += 1

                    # Validate
                    if self.trainstep % self.valid_steps == 0:
                        if self.no_valid == False:
                            valid_iter = build_dataset_iter(lazily_load_dataset('valid', self.opt),
                                                        self.fields, self.opt)
                            valid_stats = self.validate(valid_iter)

                    # Create checkpoint
                    if self.trainstep % 5000 == 0:
                        self.trainer.model_saver._save(self.trainstep)
            
            # Train on remaining partial batch
            if len((self.similar_pairs.pairs)) > 0:
                train_stats = self.trainer.train(self.similar_pairs.yield_batch())
                self.trainstep += 1

        # Write epoch statistics
        self.write_similarities(epoch_similarities, 'e{}_comp'.format(self.trainer.cur_epoch))
        self.write_similarities(epoch_scores, 'e{}_comp_scores'.format(self.trainer.cur_epoch))
        self.trainer.report_epoch()
        self.logger.info('Accepted parrallel sentences from comparable data: %d / %d' %
                    (self.accepted, self.total))
        self.logger.info('Acceptable parrallel sentences from comparable data (out of limit): %d / %d' %
                    (self.accepted_limit, self.total))
        self.logger.info('Declined sentences from comparable data: %d / %d' %
                    (self.declined, self.total))

        # Reset epoch statistics
        self.accepted = 0
        self.accepted_limit = 0
        self.declined = 0
        self.total = 0
        self.accepted_file.close()
        return train_stats

    def validate(self, valid_iter):
        return self.trainer.validate(valid_iter)

