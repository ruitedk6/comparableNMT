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
# import onmt.io
# import onmt.modules
from onmt.trainer import Trainer

import math
from collections import defaultdict
import numpy as np
from sklearn.mixture import GaussianMixture as GMM 
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


    def add_example(self, batch, ex):
        """ Add an example from a batch to the PairBank (self.pairs).
        Args:
            batch(torchtext.data.batch.Batch): batch containing the pair
            ex(int): index of the example
        """
        # Get example from src/tgt and remove original padding
        src = PairBank.removePadding(batch.src[0][:, ex])
        tgt = PairBank.removePadding(batch.tgt[:, ex])
        src_length = batch.src[1][ex]
        index = batch.indices[ex]
        # Create CompExample object holding all information needed for later
        # batch creation.
        example = CompExample(batch.dataset, batch.fields, src, tgt,
                              src_length, index)
        self.pairs.append(example)
        self.index_memory.add(hash((str(src), str(tgt))))
        return None

    def contains_batch(self):
        """Check if enough parallel pairs found to create a batch.
        """
        return (len(self.pairs) >= self.batch_size)

    def no_limit_reached(self, batch, ex):
        src = PairBank.removePadding(batch.src[0][:, ex])
        tgt = PairBank.removePadding(batch.tgt[:, ex])
        return (hash((str(src), str(tgt))) in self.index_memory or len(self.index_memory) < self.limit)
# 
    def get_max_length_sequence(examples):
        return max([ex.size(0) for ex in examples])

    def shape_example(example, max_len):
        """ Formats an example to fit with its batch.
        Args:
            example(torch.Tensor): a src/tgt sequence (size(seq))
        """
        # Add batch dimension
        example = example.unsqueeze(1)
        # Pad to max_len if necessary
        pad_size = max_len - example.size(0)
        if pad_size != 0:
            #if torch.cuda.is_available():
            #    print("here")
            #    pad = torch.ones(pad_size, 1, dtype=torch.long).cuda()
            #else
            pad = torch.ones(pad_size, 1, dtype=torch.long)
            example = torch.cat((example, pad), 0)
        return example

    def preprocess_side(examples):
        """ Formats a list of examples into one tensor.
        Args:
            examples(list): list of src/tgt sequence tensors
        Returns:
            batch(torch.Tensor): src/tgt side of the batch (size(seq, batch))
        """
        max_len = PairBank.get_max_length_sequence(examples)
        examples = [PairBank.shape_example(ex, max_len) for ex in examples]
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
# 
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
        src = PairBank.preprocess_side(src_examples)
        tgt = PairBank.preprocess_side(tgt_examples)
        src_lengths = torch.cat([length.unsqueeze(0) for length in src_lengths])
        indices = torch.cat([index.unsqueeze(0) for index in indices])

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
# 
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
        if batch % self.opt.report_every == -1 % self.opt.report_every:
            t = report_stats.elapsed_time()
            #TODO: Figure out why reported acc, ppl etc. so bad, and why num tgt words = 0
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
        # Set gradients to zero
        #self.model.zero_grad()
        # Initialize
        #self.train_loss.cur_dataset = batch.dataset
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
        """ Reports statistics of current epoch."""
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
# 
#     def epoch_step(self, ppl):
#         """ Decays learning rate if needed.
#         Args:
#             ppl(float): perplexity on validation data
# 
#         Returns:
#             decay(boolean): whether a decay has taken place or not
#         """
#         decay = super(CompTrainer, self).epoch_step(ppl, self.cur_epoch)
#         if decay:
#             self.logger.info("Decaying learning rate to %g" % self.optim.lr)
#         return decay
# 
class Comparable():
    """
    Class that controls the extraction of parallel sentences and manages their
    storage and training.

    Args:
        comp_iter(:obj:'train.DatasetLazyIter'): comparable data iterator
        model(:py:class:'onmt.Model.NMTModel'):
            translation model used for extraction and training
        trainer(:obj:'onmt.Trainer.Trainer'): 
            trainer that controlls the training process
        fields(dict): fields and vocabulary
        logger(logging.RootLogger):
            logger that reports information about extraction and training
        report_func(fn): function for logging
        opt(argparse.Namespace): option object
    """

    def __init__(self, model, trainer, fields, logger, opt):
        self.encoder = model.encoder
        self.sim_measure = opt.sim_measure
        self.threshold = opt.threshold
        self.similar_pairs = PairBank(opt.batch_size, opt)
        self.trainer = CompTrainer(trainer, logger, opt)
        self.fields = fields
        self.logger = logger
        self.accepted = 0
        self.accepted_limit = 0
        self.declined = 0
        self.total = 0
        self.comp_log = opt.comp_log
        self.cove_type = opt.cove_type
        self.em_prob_threshold = opt.em_prob_threshold
        self.percentile = opt.percentile

    def replaceEOS(fets):
        """Gets rid of EOS and SOS.
        Args:
            fets(torch.Tensor): src/tgt tensor (size(seq, batch, 1))
        Note: This only works when EOS==3!
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
            fets = Comparable.replaceEOS(fets)
        if side == 'src':
            _, lengths = batch.src
        return fets, lengths

    def forward(self, side):
        """ F-prop a src or tgt batch through the encoder.
        Args:
            side(torch.Tensor): batch to be f-propagated
                (size(seq, batch, 1))

        Returns:
            memory_bank(torch.Tensor): encoder outputs
                (size(seq, batch, fets))
        """
        with torch.no_grad():
            enc_states, memory_bank, src_lengths = self.encoder(side, None)
            return memory_bank

    def calculate_similarity(self, src, tgt):
        """ Calculates the similarity between two sentence representations.
        Args:
            src(torch.Tensor): src sentence representation (size(fets))
            tgt(torch.Tensor): tgt sentence representation (size(fets))
        """
        if self.sim_measure == "cosine":
            # tolist() here only retrieves the scalar (no list)
            return nn.functional.cosine_similarity(src, tgt, dim=0).tolist()
        else:
            return None

    def idx2words(self, seq, side):
        """ Convert word indices to words.
        Args:
            seq(torch.tensor): a src/tgt sequence (size(seq))
            side(str): {'src'|'tgt'}
        """
        vocab = self.fields[side].vocab.itos
        words = [vocab[idx] for idx in seq.data.tolist() \
                 if idx not in [0, 1, 2, 3]]
        return words

    def write_sentence(self, src, tgt, ex, status, sim=None):
        """
        Writes an accepted/rejected parallel sentence candidate pair to a file.

        Args:
            src(torch.tensor): src batch (size(seq, batch, fets))
            tgt(torch.tensor): tgt batch (size(seq, batch, fets))
            ex(int): index of the example in src/tgt batches
            status(str): ['accepted', 'accepted-limit', 'rejected']
            sim(float): similaritiy of the sentence pair
        """ 
        src_words = self.idx2words(src[:, ex, 0], 'src')
        tgt_words = self.idx2words(tgt[:, ex, 0], 'tgt')
        out = 'src: {}\ttgt: {}\tsimilarity: {}\tstatus: {}\n'.format(' '.join(src_words), 
                                            ' '.join(tgt_words), sim, status)
        if status == 'accepted' or status == 'accepted-limit':
            self.accepted_file.write(out)
        else:
            self.rejected_file.write(out)
        return None

    def get_similarities(self, src, tgt):
        """
        Prepare a batch of src and tgt pairs and calculate their similarities.
        Args:
            src(torch.tensor): src batch (size(seq, batch, 1))
            tgt(torch.tensor): src batch (size(seq, batch, 1))
        Returns:
            similarities(list): a list of similarities (float)
        """
        # Get context vectors of src and tgt
        src_memory = self.forward(src)
        tgt_memory = self.forward(tgt)

        similarities = []

        for ex in range(src.size(1)):
            # Get current example
            src_ex = src_memory[:, ex, :]
            tgt_ex = tgt_memory[:, ex, :]
            # Get sentence representation 
            if self.cove_type == "mean":
                src_cove = torch.mean(src_ex, dim=0)
                tgt_cove = torch.mean(tgt_ex, dim=0)
            else: 
                src_cove = torch.sum(src_ex, dim=0)
                tgt_cove = torch.sum(tgt_ex, dim=0)
            # Calculate similarity
            sim = self.calculate_similarity(src_cove, tgt_cove)
            similarities.append(sim)

        return similarities
# 
    def extract_parallel_sents(self, batch):
        """ 
        Extracts parallel sentences from a batch and adds them to the
        PairBank.

        Args:
            batch(torchtext.data.batch.Batch): batch object
        """
        # Get src and tgt
        src, src_lengths = Comparable.getFeatures(batch, 'src')
        tgt, _ = Comparable.getFeatures(batch, 'tgt')

        similarities = self.get_similarities(src, tgt)
        #for ex in range(len(similarities)):
        #    print(similarities[ex])
        #    print(self.idx2words(src[:, ex, 0], 'src'))
        #    print(self.idx2words(tgt[:, ex, 0], 'tgt'))

        assert (len(similarities) == src.size(1)), \
            "Number of similarities != number of examples in batch!"

        for ex in range(src.size(1)):
            if similarities[ex] >= self.threshold:
                if self.similar_pairs.no_limit_reached(batch, ex):
                    # Add to PairBank if similarity above threshold
                    self.similar_pairs.add_example(batch, ex)
                    self.accepted += 1
                    self.write_sentence(src, tgt, ex, 'accepted', similarities[ex])
                else:
                    self.accepted_limit += 1
                    self.write_sentence(src, tgt, ex, 'accepted-limit', similarities[ex])

            else:
                self.declined +=1
                self.write_sentence(src, tgt, ex, 'rejected', similarities[ex])
            self.total += 1

        return similarities
 
    def write_similarities(self, values, name):
        val_count = defaultdict(int)
        for val in values:
            val_count[round(val, 2)] += 1

        with open("{}_{}_similarities.tsv".format(self.comp_log, name), 'w+', encoding='utf8') as out:
            for val in list(val_count.keys()):
                out.write("{}\t{}\n".format(val, val_count[val]))


    def em(self, values):
        """
        Uses expectation maximization over the comparable data's similarity distribution
        to infer an appropriate threshold.
        """
        values = np.expand_dims(np.array(values), axis=1)
        means = np.array([[0.0], [0.4]])
        gmm = GMM(n_components=2, covariance_type='full', means_init=means).fit(values)
        values_sorted = np.sort(values, axis=0)
        probs = gmm.predict_proba(values_sorted)

        for sim, true_prob in zip(np.squeeze(values_sorted, axis=1).tolist(), probs[:, 1].tolist()):
            if true_prob >= self.em_prob_threshold:
                return sim

    def infer_threshold(self, data_iter, dataset, infer_type):
        """
        Gets the mean and standard deviation of the similarities of the
        training data to infer an appropriate threshold.
        Args:
            data_iter(:obj:'train.DatasetLazyIter'): data iterator
            dataset(str): dataset type ['base'|'comp']
            infer_type(str): type of inference ['mean-s'|'mean'|'mean+s']
        Return:
            threshold(float): the threshold set for sentence extraction
        """
        items = 0
        values = []
        for batch in data_iter:
            src, src_lengths = Comparable.getFeatures(batch, 'src')
            tgt, _ = Comparable.getFeatures(batch, 'tgt')
            similarities = self.get_similarities(src, tgt)
            items += len(similarities)
            values += similarities
        
        self.estim_values = values
        
        if infer_type in ['mean-s', 'mean', 'mean+s']:
            self.write_similarities(values, 'estim_{}'.format(dataset))
            mean = sum(values) / items
            variance = sum([math.pow((value - mean), 2) for value in values]) / items
            s_d = math.sqrt(variance)
            
            if infer_type == 'mean-s':
                threshold = mean - s_d
            elif infer_type == 'mean':
                threshold = mean
            else:
                threshold = mean + s_d
        
        elif infer_type == 'percentile':
            threshold = np.percentile(np.array(values), self.percentile, axis=0)
        
        elif infer_type == 'em':
            threshold = self.em(values)

        self.threshold = threshold
        self.logger.info('Threshold for comparable training is set to: %.3f' %
                         threshold)
        return threshold


    def update_threshold(self, dynamics, threshold_type):
        if dynamics == 'decay':
            update = -0.01
        else:
            update = 0.01

        if threshold_type == 'em':
            if self.em_prob_threshold < 0.95:
                self.em_prob_threshold += update
                self.threshold = self.em(self.estim_values)
        elif threshold_type in ['mean-s', 'mean', 'mean+s']:
            if self.threshold < 0.95:
                self.threshold += update
        elif threshold_type == 'percentile':
            if self.percentile < 99:
                self.percentile += (update * 100)
                self.threshold = np.percentile(np.array(self.estim_values), self.percentile)

        self.logger.info('Threshold for comaprable training is set to: %.3f' %
                                 self.threshold)




    def extract_and_train(self, comp_iter):
        """ Manages the alternating extraction of parallel sentences and training.
        Returns:
            train_stats(:obj:'onmt.Trainer.Statistics'): epoch loss statistics
        """
        # Start first epoch
        self.trainer.next_epoch()
        self.accepted_file = \
                open('{}_accepted-e{}.txt'.format(self.comp_log, self.trainer.cur_epoch), 'w+', encoding='utf8')
        self.rejected_file = \
                open('{}_rejected-e{}.txt'.format(self.comp_log, self.trainer.cur_epoch) ,'w+', encoding='utf8')
        epoch_similarities = []
        # Go through comparable data
        for batch in comp_iter:
            # Extract parallel sentences from batch
            similarities = self.extract_parallel_sents(batch)
            epoch_similarities += similarities
            # Check if enough parallel sentences were collected
            while self.similar_pairs.contains_batch():
                # Get a batch of extracted parrallel sentences and train
                training_batch = self.similar_pairs.yield_batch()
                train_stats = self.trainer.train(training_batch)
         # Train on remaining partial batch
        if len((self.similar_pairs.pairs)) > 0:
            train_stats = self.trainer.train(self.similar_pairs.yield_batch())

        self.write_similarities(epoch_similarities, 'e{}_comp'.format(self.trainer.cur_epoch))
        self.trainer.report_epoch()
        self.logger.info('Accepted parrallel sentences from comparable data: %d / %d' %
                    (self.accepted, self.total))
        self.logger.info('Acceptable parrallel sentences from comparable data (out of limit): %d / %d' %
                    (self.accepted_limit, self.total))
        self.logger.info('Declined sentences from comparable data: %d / %d' %
                    (self.declined, self.total))
        self.accepted = 0
        self.accepted_limit = 0
        self.declined = 0
        self.total = 0
        self.accepted_file.close()
        self.rejected_file.close()
        return train_stats

    def validate(self, valid_iter):
        return self.trainer.validate(valid_iter)

