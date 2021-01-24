import array
import numpy as np
import random
import math
import os
import logging
import time
import argparse


class Vocabulary(object):

    def __init__(self, path, min_occurrences=500, build=False):
        self._tok2id = {}
        self._id2tok = {}
        self._n_occurrences = {}
        self.unk = '<UNK>'
        self.pad = '<PAD>'
        self.mask = '<MASK>'
        self.bos = '<S>'
        self.eos = '</S>'
        self._unk_id = -1
        self._pad_id = -1
        self._mask_id = -1
        self._bos_id = -1
        self._eos_id = -1

        if build:
            self._build(path)
        else:
            self._load(path)

        if self._unk_id == -1:
            self._unk_id = self._allocate_id(self.unk)
        if self._pad_id == -1:
            self._pad_id = self._allocate_id(self.pad)
        if self._mask_id == -1:
            self._mask_id = self._allocate_id(self.mask)
        if self._bos_id == -1:
            self._bos_id = self._allocate_id(self.bos)
        if self._eos_id == -1:
            self._eos_id = self._allocate_id(self.eos)

        self._vocab = set(
                [tok for tok, n_occurrences in self._n_occurrences.items()
                 if n_occurrences >= min_occurrences])

        for n_occurrences in self._n_occurrences.values():
            if n_occurrences < min_occurrences:
                self._n_occurrences[self._unk_id] += n_occurrences
        
        self._vocab.add(self._unk_id)
        self._vocab.add(self._pad_id)
        self._vocab.add(self._mask_id)
        self._vocab.add(self._bos_id)
        self._vocab.add(self._eos_id)
        
        # sort vocab so that most frequent tokens come first
        self._vocab_list = sorted(
                list(self._vocab), key=lambda t: (-self._n_occurrences[t], t))
        self.size = len(self._vocab)
        
        self._tokid2vocabid = {
                tok_id: vocab_id
                for vocab_id, tok_id in enumerate(self._vocab_list)}
        self.unk_vocab_id = self._tokid2vocabid[self._unk_id]
        self.pad_vocab_id = self._tokid2vocabid[self._pad_id]
        self.mask_vocab_id = self._tokid2vocabid[self._mask_id]
        self.bos_vocab_id = self._tokid2vocabid[self._bos_id]
        self.eos_vocab_id = self._tokid2vocabid[self._eos_id]

        self.freq = np.ones([self.size])
        for i, tok in enumerate(self._vocab_list):
            self.freq[i] += float(self._n_occurrences[tok])
        self.freq /= np.sum(self.freq)

    def _allocate_id(self, token):
        new_id = len(self._tok2id)
        self._tok2id[token] = new_id
        self._id2tok[new_id] = token
        self._n_occurrences[new_id] = 0
        return new_id

    def id2str(self, vocab_id):
        return self._id2tok[self._vocab_list[vocab_id]]

    def str2id(self, token):
        if token not in [
                self.mask, self.bos, self.eos, self.pad, self.unk]:
            token = token.lower()
        if token not in self._tok2id:
            return self.unk_vocab_id
        token_id = self._tok2id[token]
        if token_id not in self._vocab:
            return self.unk_vocab_id
        return self._tokid2vocabid[token_id]

    def sentence2str(self, sentence):
        return ' '.join([self.id2str(x) for x in sentence
                         if x != self.pad_vocab_id])

    def write_vocab_file(self, vocab_path):
        with open(vocab_path, 'w') as f:
            f.write('%d %d\n' % (len(self._tok2id), self.n_tokens))

            for i in range(len(self._tok2id)):
                n_occurrences = (0 if i == self._unk_id
                                 else self._n_occurrences[i])
                f.write('%s %d\n' % (self._id2tok[i], n_occurrences))

    def get_n_occurrences(self, vocab_id):
        return self._n_occurrences[self._vocab_list[vocab_id]]

    def _load(self, path):
        """ The format of the vocab file is as follows:
            n_unique_tokens n_tokens_total
            token_1 n_occurrences_1
            token_2 n_occurrences_2
            .
            .
            .
            token_n n_occurrences_n"""
        logging.info('Loading vocab from %s...' % path)
        with open(path, 'r') as f:
            ns = f.readline().strip().split()
            self._n_unique_tokens = int(ns[0])
            self.n_tokens = int(ns[1])
            
            for i in range(self._n_unique_tokens):
                line = f.readline().strip().split()
                if len(line) != 2:
                    logging.warn('Invalid line: ' + str(line))
                    continue
                tok = line[0]
                count = int(line[1])
                self._tok2id[tok] = i
                self._id2tok[i] = tok
                self._n_occurrences[i] = count
                if tok == self.unk:
                    self._unk_id = i
                elif tok == self.pad:
                    self._pad_id = i
                elif tok == self.mask:
                    self._mask_id = i
                elif tok == self.bos:
                    self._bos_id = i
                elif tok == self.eos:
                    self._eos_id = i

    def _build(self, path):
        self._n_unique_tokens = 0
        self.n_tokens = 0
        
        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                for tok in tokens:
                    tok = tok.lower()
                    self.n_tokens += 1
                    if self.n_tokens % 10000000 == 0:
                        logging.info(
                                'Read %d million tokens' %
                                (self.n_tokens // 1000000))
                    if tok in self._tok2id:
                        tokid = self._tok2id[tok]
                        self._n_occurrences[tokid] += 1
                    else:
                        tokid = self._n_unique_tokens
                        self._n_unique_tokens += 1
                        self._tok2id[tok] = tokid
                        self._id2tok[tokid] = tok
                        self._n_occurrences[tokid] = 1
                        if tok == self.unk:
                            self._unk_id = i
                        elif tok == self.mask:
                            self._mask_id = i
                        elif tok == self.bos:
                            self._bos_id = i
                        elif tok == self.eos:
                            self._eos_id = i

class Batch(object):

    def __init__(self, unmasked_seqs, masked_seqs, seq_len,
                 target_positions, targets, line_num=0):
        self.unmasked_seqs = unmasked_seqs
        self.masked_seqs = masked_seqs
        self.seq_len = seq_len
        self.target_positions = target_positions
        self.targets = targets
        self.line_num = line_num

    def n_tokens(self):
        return np.sum(self.seq_len - 2)

class ProtoBatch(object):

    def __init__(self, size, length, pad):
        self._size = size
        self._unmasked_seqs = pad * np.ones([size, length], dtype=np.int32)
        self._masked_seqs = pad * np.ones([size, length], dtype=np.int32)
        self._seq_len = np.zeros([size], dtype=np.int32)
        self._target_positions = []
        self._targets = []
        self._length = length
        self._count = 0
        self._line_num = 0

    def to_batch(self):
        if self._count < self._size:
            self._unmasked_seqs = self._unmasked_seqs[:self._count, :]
            self._masked_seqs = self._masked_seqs[:self._count, :]
            self._seq_len = self._seq_len[:self._count]

        return Batch(self._unmasked_seqs,
                     self._masked_seqs,
                     self._seq_len,
                     np.array(self._target_positions),
                     np.array(self._targets),
                     line_num=self._line_num)

    def add_sentence(self, s, target_indices, masks, line_num):
        self._line_num = line_num
        n = len(s)
        self._unmasked_seqs[self._count, :n] = s
        for i, index in enumerate(target_indices):
            self._target_positions.append(
                    self._count * self._length + index)
            self._targets.append(s[index])
            s[index] = masks[i]

        self._masked_seqs[self._count, :n] = s
        self._seq_len[self._count] = n

        self._count += 1

        return self._count == self._size

        
class Batcher(object):

    def __init__(self, batch_size, max_length, pad,
                 variable_length=False, pad_prob=0.1):
        self._batch_size = batch_size
        self._max_length = max_length
        self._pad = pad
        self._variable_length = variable_length
        self._pad_prob = pad_prob
        self._proto_batches = [
                    self._make_proto_batch(length)
                if self._variable_length or length == self._max_length
                else
                    None
                for length in range(self._max_length + 1)]

    def _make_proto_batch(self, length):
        return ProtoBatch(
                self._batch_size, length, self._pad)

    def add_sentence(self, s, target_indices, masks, line_num):
        if self._variable_length:
            length = len(s)
            if len(s) < self._max_length and random.random() < self._pad_prob:
                length = random.randint(length + 1, self._max_length)
        else:
            length = self._max_length

        if self._proto_batches[length].add_sentence(
                s, target_indices, masks, line_num):
            batch = self._proto_batches[length].to_batch()
            self._proto_batches[length] = self._make_proto_batch(length)
            return batch

        return None

class Corpus(object):

    def __init__(self, path, vocab):
        self._path = path
        self._vocab = vocab

    def read_lines(self, max_seq_len):
        while True:
            with open(self._path, 'r') as f:
                for i, line in enumerate(f):
                    tokens = line.strip().split()
                    if len(tokens) > max_seq_len - 2 or len(line) == 0:
                        continue
                    yield ([self._vocab.bos_vocab_id] +
                           [self._vocab.str2id(t) for t in tokens] +
                           [self._vocab.eos_vocab_id], i)

    def generate_batches(self, batch_size, max_seq_len, mask_prob,
                         must_contain=None, variable_length=False,
                         pad_prob=0.1, masking_policy=[1.0, 0.0, 0.0]):
        batcher = Batcher(batch_size, max_seq_len, self._vocab.pad_vocab_id,
                          variable_length=variable_length, pad_prob=pad_prob)
        for line, line_num in self.read_lines(max_seq_len):
            if must_contain is not None:
                valid = False
                for t in line:
                    if t == must_contain:
                        valid = True
                if not valid:
                    continue
            if mask_prob > 0.0:
                mask_values = np.random.uniform(size=[len(line) - 2])
                target_indices = np.where(mask_values < mask_prob)[0] + 1
                if len(target_indices) == 0:
                    continue
                masks = []
                assert abs(sum(masking_policy) - 1.0) < 1e-6
                mask_token_thresh = masking_policy[0]
                self_thresh = mask_token_thresh + masking_policy[1]
                for i in target_indices:
                    r = random.random()
                    if r < mask_token_thresh:
                        masks.append(self._vocab.mask_vocab_id)
                    elif r < self_thresh:
                        masks.append(line[i])
                    else:
                        while True:
                            t = random.randint(0, self._vocab.size-1)
                            if t not in [self._vocab.pad_vocab_id,
                                         self._vocab.bos_vocab_id,
                                         self._vocab.eos_vocab_id]:
                                masks.append(random.randint(0, self._vocab.size-1))
                                break
            else:
                target_indices = []
                masks = []
        
            maybe_batch = batcher.add_sentence(
                    line, target_indices, masks, line_num)
            if maybe_batch is not None:
                yield maybe_batch

