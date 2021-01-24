#import gc
import logging
import numpy as np
import os
import random
import time
import itertools
import sys

import data

from xml.etree import ElementTree
from nltk.corpus import wordnet as wn
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tag.stanford import StanfordTagger
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

np.set_printoptions(linewidth=np.inf)

PLURAL = '<PL>'
COMP = '<COMP>'
SUP = '<SUP>'
PAST = '<PAST>'
GER = '<GER>'
NONTHIRD = '<N3RD>'
THIRD = '<3RD>'
PART = '<PART>'

APPEND = {
        'NNS': PLURAL,
        'JJR': COMP,
        'JJS': SUP,
        'RBR': COMP,
        'RBS': SUP,
        'VBD': PAST,
        'VBG': GER,
        'VBP': NONTHIRD,
        'VBZ': THIRD,
        'VBN': PART
}

SUFFIXES = set(APPEND.values())

MAP = {
        'NNS': 'n',
        'JJR': 'a',
        'JJS': 'a',
        'RBR': 'r',
        'RBS': 'r',
        'VBD': 'v',
        'VBG': 'v',
        'VBP': 'v',
        'VBZ': 'v',
        'VBN': 'v'
}

def stem_focus(focus, pos):
    if focus == 'lay':
        return 'lie', PAST
    stemmed = lemmatize(focus, pos)
    if focus.endswith('ing'):
        suf = GER
    elif focus.endswith('s'):
        if pos == 'v':
            suf = THIRD
        else:
            suf = PLURAL
    elif focus.endswith('ed'):
        suf = PAST
    else:
        suf = PART
    return stemmed, suf

def stem(sentences):
    pos_tagger_root = FLAGS.pos_tagger_root
    tagger = StanfordPOSTagger(
            pos_tagger_root + '/models/english-left3words-distsim.tagger',
            path_to_jar=pos_tagger_root + '/stanford-postagger.jar')
    for s in sentences:
        s.append('\n')
    joined = list(itertools.chain.from_iterable(sentences))
    logging.info('About to dispatch to Stanford POS tagger')
    output = tagger.tag(joined)
    logging.info('Got result from Stanford POS tagger')
    #print(output)
    start = 0
    stemmed_sentences = []
    alignments = []
    tags = []
    for s in sentences:
        #logging.info(s)
        end = start + len(s) - 1
        #logging.info(output[start:end])
        tagged_sentence = output[start:end]
        stemmed_sentence = []
        alignment = []
        tag = []
        for token, pos in tagged_sentence:
            token = token.lower()
            alignment.append(len(stemmed_sentence))
            if pos in MAP and token.isalpha():
                stemmed_token = lemmatize(token, MAP[pos])
                stemmed_sentence.append(stemmed_token)
                stemmed_sentence.append(APPEND[pos])
            else:
                stemmed_sentence.append(token)
            tag.append(pos)

        stemmed_sentences.append(stemmed_sentence)
        alignments.append(alignment)
        tags.append(tag)
        start = end

    assert start == len(output)
    return stemmed_sentences, alignments, tags

class StanfordPOSTagger(StanfordTagger):

    _SEPARATOR = '_'
    _JAR = 'stanford-postagger.jar'

    def __init__(self, *args, **kwargs):
        super(StanfordPOSTagger, self).__init__(*args, **kwargs)

    @property
    def _cmd(self):
        return [
                'edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model',
                self._stanford_model,
                '-textFile',
                self._input_file_path,
                '-tokenize',
                'false',
                '-outputFormatOptions',
                'keepEmptySentences',
                '-sentenceDelimiter',
                'newline',
                '-options',
                _TOKENIZER_OPTIONS_STR
        ]

class Tokenizer(StanfordTokenizer):

    def tokenize(self, s):
        cmd = ['edu.stanford.nlp.process.PTBTokenizer', '-preserveLines']
        return self._parse_tokenized_output(self._execute(cmd, s))


_TOKEN_MAP = {"``": '"', "''": '"', '--': '-', "/?": "?", "/.": ".", '-LRB-': '(', '-RRB-': ')'}
_SEPARATOR = ' <SEP> '
_TOKENIZER_OPTIONS = {'tokenizePerLine': 'true',
                      'americanize': 'false',
                      'normalizeCurrency': 'false',
                      'normalizeParentheses': 'false',
                      'normalizeOtherBrackets': 'false',
                      'asciiQuotes': 'false',
                      'latexQuotes': 'false',
                      'unicodeQuotes': 'false',
                      'ptb3Ellipsis': 'false',
                      'unicodeEllipsis': 'false',
                      'ptb3Dashes': 'false',
                      'splitHyphenated': 'true'}
_TOKENIZER_OPTIONS_STR = ','.join(
        ['%s=%s' % item for item in _TOKENIZER_OPTIONS.items()])
_TOKENIZER = None

def get_tokenizer():
    return Tokenizer(
            path_to_jar=FLAGS.pos_tagger_root + '/stanford-postagger.jar',
            options=_TOKENIZER_OPTIONS)

_LEMMATIZER = WordNetLemmatizer()

_NLTK_TO_WORDNET_POS = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 's': 'ADJ', 'r': 'ADV'}
_WORDNET_TO_NLTK_POS = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}

def lemmatize(word, pos):
    return _LEMMATIZER.lemmatize(word, pos)

def map_token(token):
    return _TOKEN_MAP.get(token, token)

def preprocess_str(s):
    return ' '.join([map_token(t) for t in s.split()])

def split_token(token, vocab):
    token = map_token(token.lower())
    subtokens = []
    for s in token.split():
        if vocab.str2id(s) != vocab.unk_vocab_id:
            subtokens.append(s)
        else:
            t = s.split('-')
            for i, u in enumerate(t):
                if i > 0:
                    subtokens.append('-')
                if u:
                    subtokens.append(u)
    return subtokens

def tokenize_sequence(seq, vocab):
    return list(itertools.chain.from_iterable(
            [split_token(t, vocab) for t in seq]))
    
def tokenize(seqs, tokenizer):
    tokenized_seqs = tokenizer.tokenize('\n'.join(seqs))
    tokenized_seqs = [s.split() for s in tokenized_seqs]
    alignments = []
    for seq, tok in zip(seqs, tokenized_seqs):
        alignment = []
        i = 0
        build = ''
        for word in seq.split():
            while True:
                build += tok[i]
                i += 1
                if build == word:
                    alignment.append(i-1)
                    build = ''
                    break
                if i >= len(tok):
                    logging.error('Bad tokenization:\n%s\n%s' %
                            (seq, ' '.join(tok)))
                    break
                    #sys.exit(0)
            if i >= len(tok):
                break
        alignments.append(alignment)
    tokenized_seqs = [[map_token(t) for t in s] for s in tokenized_seqs]
    return tokenized_seqs, alignments

def wordnet_pos(nltk_lemma):
    return _NLTK_TO_WORDNET_POS[nltk_lemma.synset().pos()]

def make_span(vocab_ids, index, vocab, max_seq_length):
    before = (max_seq_length - 2) // 2
    after = max_seq_length - 2 - before - 1
    left_space = index
    right_space = len(vocab_ids) - index - 1
    extra_left = max(0, after - right_space)
    extra_right = max(0, before - left_space)
    start = max(0, index - before - extra_left)
    end = min(len(vocab_ids), index + after + extra_right + 1)
    assert end - start + 2 <= max_seq_length
    span = [vocab.bos_vocab_id] + vocab_ids[start:end] + [vocab.eos_vocab_id]
    index_in_span = index - start + 1
    return span, index_in_span

def display_word(corpus, word, similarities, tokens, senses, n_senses,
                 info=None):
    vocab_id = corpus.str2id(word)
    for sense in range(n_senses):
        neighbour_strs = [
                '%s_%d(%.3f)' % (corpus.id2str(t), s+1, c)
                for c, t, s in zip(similarities[sense, :],
                                   tokens[sense, :],
                                   senses[sense, :])]
        info_str = '' if info is None else ' (%s)' % info[sense]
        logging.info('%s_%d%s: %s' % (
                word, sense + 1, info_str, ' '.join(neighbour_strs)))

def to_sentence(vocab, vocab_ids, highlight=None):
    def to_str(vocab_id, index):
        word = vocab.id2str(vocab_id)
        if highlight is not None and index == highlight:
            word = '**%s**' % word
        return word

    return ' '.join([to_str(t, i) for i, t in enumerate(vocab_ids)])

def make_batch(instances, vocab, max_seq_len,
               mask=True, replace_with_lemma=True):
    n_instances = len(instances)
    ids = vocab.pad_vocab_id * np.ones(
            [n_instances, max_seq_len], dtype=np.int32)
    seq_len = np.zeros([n_instances], dtype=np.int32)
    masked_indices = np.zeros([n_instances, 2], dtype=np.int32)
    masked_ids = np.zeros([n_instances], dtype=np.int32)
    unmasked_seqs = vocab.pad_vocab_id * np.ones(
            [n_instances, max_seq_len], dtype=np.int32)
    masked_seqs = vocab.pad_vocab_id * np.ones(
            [n_instances, max_seq_len], dtype=np.int32)
    target_positions = np.zeros([n_instances], dtype=np.int32)
    targets = np.zeros([n_instances], dtype=np.int32)

    for i, instance in enumerate(instances):
        vocab_ids = [vocab.str2id(t) for t in instance['tokens']]
        span, index = make_span(
                vocab_ids, instance['index_in_seq'], vocab, max_seq_len)
        instance['span'] = span
        instance['index_in_span'] = index
        unmasked_seqs[i, :len(span)] = span
        masked_seqs[i, :len(span)] = span
        if mask:
            masked_seqs[i, index] = vocab.mask_vocab_id
        seq_len[i] = len(span)
        target_positions[i] = i * max_seq_len + index
        if replace_with_lemma:
            targets[i] = vocab.str2id(instance['lemma'])
        else:
            targets[i] = span[index]

    return data.Batch(
            unmasked_seqs, masked_seqs, seq_len, target_positions, targets)

class WicCorpus(object):

    def __init__(self, examples, vocab, stem=False):
        self._vocab = vocab
        self._lemmas = []
        self._sentences = []
        self._focuses = []
        self._pos = []
        seqs = []

        for sentence, focus, lemma, pos in examples:
            preprocessed = preprocess_str(sentence)
            self._lemmas.append(lemma)
            self._sentences.append(preprocessed)
            self._focuses.append(focus)
            self._pos.append(pos)
            seqs.append(sentence)

        token_lists, alignments = tokenize(seqs, get_tokenizer())
        self._tokens = []
        for i, (tokens, alignment) in enumerate(zip(token_lists, alignments)):
            self._focuses[i] = alignment[self._focuses[i]]
            self._tokens.append(tokens)

        if stem:
            self.stem()

    def stem(self):
        logging.info('Stemming WiC corpus')
        stemmed, alignments, tags = stem(self._tokens)
        logging.info('Finished stemming corpus')
        for i, alignment in enumerate(alignments):
            self._focuses[i] = alignment[self._focuses[i]]
            toks = stemmed[i]
            ind = self._focuses[i]
            stemmed_focus = toks[ind]
            if stemmed_focus != self._lemmas[i]:
                new_stem, suf = stem_focus(stemmed_focus, self._pos[i])
                if new_stem == self._lemmas[i]:
                    stemmed[i] = toks[:ind] + [new_stem, suf] + toks[ind+1:]
                    logging.info('fixed %s -> %s' % (' '.join(toks), ' '.join(stemmed[i])))
                else:
                    logging.warn('stemmed focus "%s" does not match lemma "%s" in sentence "%s"' % (
                             stemmed_focus, self._lemmas[i], ' '.join(stemmed[i])))
        self._tokens = []
        for seq in stemmed:
            self._tokens.append(seq)

    def generate_instances(self):
        for tokens, focus, lemma, pos, sentence in zip(
                self._tokens, self._focuses, self._lemmas,
                self._pos, self._sentences):
            yield {'tokens': tokens, 'index_in_seq': focus,
                   'lemma': lemma, 'pos': pos, 'sentence': sentence}

    def _generate_batches(self, max_batch_size, max_seq_len):
        instance_gen = self.generate_instances()
        done = False
        while not done:
            instances = []
            while len(instances) < max_batch_size:
                try:
                    instance = next(instance_gen)
                    instances.append(instance)
                except StopIteration:
                    done = True
                    break
            yield instances, make_batch(instances, self._vocab, max_seq_len)

    def calculate_representations(
            self, sess, model, max_batch_size, max_seq_len,
            calculate_sense_probs=True,
            method='prediction'):
        all_instances = []
        for instances, batch in self._generate_batches(
                max_batch_size, max_seq_len):
            representations = model.contextualize(sess, batch)
            if calculate_sense_probs:
                sense_probs = model.disambiguate(sess, batch, method=method)
            for i, instance in enumerate(instances):
                instance['representation'] = representations[i, :]
                assert len(instance['representation'].shape) == 1
                if calculate_sense_probs:
                    instance['sense_probs'] = sense_probs[i, :]
                all_instances.append(instance)

        return all_instances

def wic(model, vocab, sess, options):
    labels = []
    if options.wic_train:
        label_map = {'F': 0, 'T': 1}
        with open(options.wic_gold_path, 'r') as gold:
            for line in gold:
                labels.append(label_map[line.strip()])

    examples = [[], []]
    with open(options.wic_data_path, 'r') as data:
        for line in data:
            parts = line.strip().split('\t')
            lemma = parts[0]
            pos = parts[1].lower()
            indices = [int(x) for x in parts[2].split('-')]
            sentences = parts[3:]
            for i, sentence in enumerate(sentences):
                examples[i].append((sentence, indices[i], lemma, pos))

    sets = [WicCorpus(e, vocab, stem=options.lemmatize)
            for e in examples]

    instances = []
    for i, dataset in enumerate(sets):
        instances.append(dataset.calculate_representations(
                sess, model, options.batch_size,
                options.max_seq_len,
                method=options.sense_prob_source))

    n_instances = len(instances[0])
    logging.info('%d examples' % n_instances)
    similarities = np.zeros([n_instances, 2])
    for i, (e1, e2) in enumerate(zip(*instances)):
        r1 = e1['representation']
        r2 = e2['representation']
        norm1 = np.linalg.norm(r1)
        norm2 = np.linalg.norm(r2)
        cosine_sim = np.dot(r1, r2) / (norm1 * norm2)

        p1 = e1['sense_probs']
        p2 = e2['sense_probs']
        prob_score = np.dot(p1, p2)
        prob_str_1 = '(%s)' % ('/'.join(['%.4f' % x for x in p1]))
        prob_str_2 = '(%s)' % ('/'.join(['%.4f' % x for x in p2]))

        assert e1['lemma'] == e2['lemma']
        model.display_words(sess, [e1['lemma']])
        logging.info(' '.join(e1['tokens']))
        logging.info(' '.join(e2['tokens']))
        label_str = ' (%d)' % labels[i] if options.wic_train else ''
        processed_sentence_1 = to_sentence(
                vocab, e1['span'], highlight=e1['index_in_span'])
        processed_sentence_2 = to_sentence(
                vocab, e2['span'], highlight=e2['index_in_span'])
        logging.info('%.4f/%.4f%s:\n%s %s\n%s %s' % (
                prob_score, cosine_sim, label_str,
                prob_str_1, processed_sentence_1,
                prob_str_2, processed_sentence_2))

        similarities[i, 0] = cosine_sim
        similarities[i, 1] = prob_score

    if options.wic_use_contextualized_reps and options.wic_use_sense_probs:
        if options.wic_train:
            classifier = LogisticRegression()
            classifier.fit(similarities, np.array(labels, dtype=np.float32))
            joblib.dump(classifier, options.wic_model_path)
        else:
            classifier = joblib.load(options.wic_model_path)
        logging.info('Coefficients: %s' % str(classifier.coef_))
        predicted_labels = classifier.predict(similarities)
        predicted_labels = [int(x) for x in predicted_labels]
    else:
        if options.wic_use_contextualized_reps:
            similarities = similarities[:, 0]
        elif options.wic_use_sense_probs:
            similarities = similarities[:, 1]
        else:
            logging.error(
                    'Must use either contextualized representations or sense '
                    'probabilities (or both) for WiC classifictation.')
            return

        median = np.median(similarities)
        if options.wic_classification_threshold == -1.0:
            logging.info('Setting classification threshold to median value')
            thresh = median
        else:
            thresh = options.wic_classification_threshold

        predicted_labels = [0 if s < thresh else 1 for s in similarities]

        logging.info('Median similarity: %.4f' % median)
        logging.info('Classification threshold: %.4f' % thresh)

    if options.wic_train:
        confusion = [[0, 0], [0, 0]]
        for i, (label, pred) in enumerate(zip(labels, predicted_labels)):
            confusion[label][pred] += 1
        n_correct = confusion[0][0] + confusion[1][1]
        n_positive = confusion[1][0] + confusion[1][1]
        logging.info('Accuracy: %.4f' % (n_correct / n_instances))
        most_common = max(n_positive, n_instances - n_positive)
        logging.info('Baseline accuracy: %.4f' % (most_common / n_instances))
        logging.info('\n' + str(np.array(confusion)))
    else:
        with open('output.txt', 'w') as f:
            for label in predicted_labels:
                f.write('%s\n' % ['F', 'T'][label])

def get_usage_examples(lemma, pos, vocab):
    instances = []
    nltk_pos = _WORDNET_TO_NLTK_POS[pos]
    for lem in wn.lemmas(lemma):
        if wordnet_pos(lem) != pos:
            continue
        key = lem.key()
        examples = lem.synset().examples()
        if len(examples) == 0:
            continue
        #examples = _SEPARATOR.join(examples)
        #tokenized = _TOKENIZER.tokenize(examples)
        #tokenized = ' '.join(tokenized)
        #tokenized = tokenized.split(_SEPARATOR)
        for i, example in enumerate(examples):
            tokenized = example.split()
            #tokenized = word_tokenize(example)
            #tokenized = tokenize_sequence(tokenized, vocab)
            #example = example.split()
            lemma_index = -1
            for j, token in enumerate(tokenized):
                token_lemma = lemmatize(token.lower(), nltk_pos)
                if token_lemma == lemma:
                    lemma_index = j
                    break

            if lemma_index != -1:
                wsd_id = '%s ex. %d' % (key, i + 1)
                instances.append({
                        'tokens': tokenized,
                        'name': wsd_id,
                        'lemma': lemma,
                        'pos': pos,
                        'index_in_seq': lemma_index,
                        'sense_keys': [key]})

    return instances

def get_definition_instances(lemma, pos, vocab):
    instances = []
    nltk_pos = _WORDNET_TO_NLTK_POS[pos]
    for lem in wn.lemmas(lemma):
        if wordnet_pos(lem) != pos:
            continue
        key = lem.key()
        definition = lem.synset().definition()
        tokenized = definition.split()
        #tokenized = word_tokenize(definition)
        #tokenized = list(itertools.chain.from_iterable(
        #        [split_token(t, vocab) for t in tokenized]))
        tokenized = ['"', lemma, '"', 'means', '"'] + tokenized + ['"']
        wsd_id = '%s def.' % key
        instances.append({
                'tokens': tokenized,
                'name': wsd_id,
                'lemma': lemma,
                'pos': pos,
                'index_in_seq': 1,
                'sense_keys': [key]})

    return instances


class WsdText(object):

    def __init__(self, xml, vocab, gold={}, stem=False, extra=None):
        self._vocab = vocab
        self._tokens = []

        self._pos = []
        self._lemmas = []
        self._focuses = []
        self._instance_names = []
        self._sense_keys = []

        if extra:
            self._tokens = extra['tokens']
            self._pos.append(extra['pos'])
            self._lemmas.append(extra['lemma'])
            self._focuses.append(extra['index_in_seq'])
            self._instance_names.append(extra['name'])
            self._sense_keys.append(extra['sense_keys'])
        else:
            for sentence in xml:
                assert sentence.tag == 'sentence'
                for token in sentence:
                    for i, t in enumerate(split_token(token.text, vocab)):
                        if i == 0 and token.tag == 'instance':
                            self._focuses.append(len(self._tokens))
                            inst_name = token.attrib['id']
                            self._instance_names.append(inst_name)
                            self._sense_keys.append(gold.get(inst_name, [None]))
                            self._pos.append(token.attrib['pos'])
                            self._lemmas.append(token.attrib['lemma'])
                        self._tokens.append(t)

    def stem(self, stemmed, alignment):
        offset = 0
        for i in range(len(self._focuses)):
            focus_index = self._focuses[i]
            ind = alignment[focus_index] + offset
            self._focuses[i] = ind
            stemmed_focus = stemmed[ind]
            if stemmed_focus != self._lemmas[i]:
                new_stem, suf = stem_focus(stemmed_focus, _WORDNET_TO_NLTK_POS[self._pos[i]])
                if new_stem == self._lemmas[i]:
                    stemmed = stemmed[:ind] + [new_stem, suf] + stemmed[ind+1:]
                    offset += 1
                    logging.warn('fixed %s -> %s %s' % (stemmed_focus, new_stem, suf))
                elif '_' not in self._lemmas[i] and '-' not in self._lemmas[i]:
                    logging.warn('%s: stemmed focus "%s" does not match lemma "%s"' % (
                            self._instance_names[i], stemmed_focus, self._lemmas[i]))
        self._tokens = stemmed

    def generate_instances(self, req_lemma=None, req_pos=None):
        for focus, lemma, pos, name, keys in zip(
                self._focuses, self._lemmas, self._pos,
                self._instance_names, self._sense_keys):
            if ((req_lemma is None or lemma == req_lemma) and
                    (req_pos is None or pos == req_pos)):
                yield {'tokens': self._tokens, 'index_in_seq': focus, 'name': name,
                       'lemma': lemma, 'pos': pos, 'sense_keys': keys}

    def get_all_lemmas(self):
        lemmas = set()
        for lemma in zip(self._lemmas, self._pos):
            lemmas.add(lemma)
        return lemmas

def get_labels(path):
    labels = {}
    if path is not None:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                labels[parts[0]] = parts[1:]
    return labels

class WsdCorpus(object):
    
    def __init__(self, path, vocab, gold_path=None, stem=False, extras=[]):
        self._vocab = vocab
        gold = {}
        if gold_path is not None:
            gold = get_labels(gold_path)

        corpus = ElementTree.parse(path).getroot()
        assert corpus.tag == 'corpus'

        self._texts = []
        for text in corpus:
            assert text.tag == 'text'
            self._texts.append(WsdText(text, vocab, gold=gold, stem=stem))

        for extra in extras:
            instances = (get_definition_instances(extra[0], extra[1], vocab) +
                         get_usage_examples(extra[0], extra[1], vocab))
            for instance in instances:
                self._texts.append(WsdText(None, vocab, stem=stem, extra=instance))
        
        if stem:
            self.stem()

    def stem(self):
        logging.info('Stemming WSD corpus')
        stemmed, alignments, _ = stem([text._tokens for text in self._texts])
        logging.info('Finished stemming corpus')
        for i, text in enumerate(self._texts):
            text.stem(stemmed[i], alignments[i])

    def _generate_instances(
            self, options, training=False, req_lemma=None, req_pos=None):
        keys = {}
        if not training or options.wsd_use_training_data:
            for text in self._texts:
                for instance in text.generate_instances(
                        req_lemma=req_lemma, req_pos=req_pos):
                    if (options.wsd_ignore_semeval_07 and
                            instance['name'].startswith('semeval2007')):
                        continue
                    if (keys.setdefault(instance['sense_keys'][0], 0) <
                            options.wsd_max_examples_per_sense or not training):
                        keys[instance['sense_keys'][0]] += 1
                        yield instance
        #if training:
        #    if options.wsd_use_usage_examples:
        #        for instance in get_usage_examples(
        #                req_lemma, req_pos, self._vocab):
        #            yield instance
        #    if options.wsd_use_definitions:
        #        for instance in get_definition_instances(
        #                req_lemma, req_pos, self._vocab):
        #            yield instance

    def _generate_batches(self, options, training=False,
                          req_lemma=None, req_pos=None):
        instance_gen = self._generate_instances(
                options, training=training,
                req_lemma=req_lemma, req_pos=req_pos)
        done = False
        while not done:
            instances = []
            while len(instances) < options.batch_size:
                try:
                    instance = next(instance_gen)
                    instances.append(instance)
                except StopIteration:
                    done = True
                    break
            yield instances, make_batch(instances, self._vocab,
                                        options.max_seq_len)

    def calculate_sense_probs(
            self, sess, model, options, training=False,
            req_lemma=None, req_pos=None,
            replace_id=None, replace_with_lemma=False,
            method='prediction'):
        assert replace_id is None or not replace_with_lemma
        all_instances = []
        for instances, batch in self._generate_batches(
                options, training=training,
                req_lemma=req_lemma, req_pos=req_pos):
            sense_probs = model.disambiguate(sess, batch, method=method)
            for i, instance in enumerate(instances):
                instance['sense_probs'] = sense_probs[i, :]
                all_instances.append(instance)

        return all_instances

    def get_all_lemmas(self):
        lemmas = set()
        for text in self._texts:
            for lemma in text.get_all_lemmas():
                lemmas.add(lemma)
        return lemmas

def get_lemmas(w):
    lemmas = wn.lemmas(w)
    return {lemma.key(): lemma for lemma in lemmas}


np.set_printoptions(precision=3, suppress=True)

def onehot(columns, n_columns):
    n_rows = len(columns)
    rows = np.arange(n_rows)
    A = np.zeros([n_rows, n_columns])
    A[rows, columns] = 1.0
    return A

CACHED_MATCH_MATRICES = {}

def match_matrix(wsd_corpus, vocab, model, sess, word, lemma, options,
                 pos=None, verbose=False):
    vocab_id = vocab.str2id(word)
    item = (vocab_id, lemma, pos)
    cached = CACHED_MATCH_MATRICES.get(item, None)
    if cached is not None:
        return cached

    n_senses = model.get_n_senses(vocab_id)
    instances = wsd_corpus.calculate_sense_probs(
            sess, model, options, training=True,
            req_lemma=lemma, req_pos=pos, replace_id=vocab_id,
            method=options.sense_prob_source)

    if (len(instances) == 0 or
            n_senses == 1 or
            '_' in lemma or
            ('-' in lemma and '-' not in word)):
        if verbose:
            logging.info('No training data for %s' % lemma)
        sense_keys = [lem.key() for lem in wn.lemmas(lemma)
                      if pos is None or wordnet_pos(lem) == pos]
        if len(sense_keys) == 0:
            logging.error('Found no senses for %s as a %s' % (lemma, pos))
        P = np.zeros([n_senses, len(sense_keys)], dtype=np.float32)
        P[:, 0] = 1.0
        CACHED_MATCH_MATRICES[item] = (P, sense_keys)
        return P, sense_keys
    
    senses = {}
    all_probs = []
    for instance in instances:
        sense_key = instance['sense_keys'][0]
        sense_probs = instance['sense_probs']
        senses.setdefault(sense_key, []).append(sense_probs)
        all_probs.append(sense_probs)
        #if verbose:
        #    model.display_words(sess, [vocab.id2str(instance['vocab_id'])])
        #    logging.info('Lemma: %s, sense key: %s' % (
        #            instance['lemma'], sense_key))
        #    logging.info('(%s)' % ('/'.join(['%.3f' % p for p in sense_probs])))
        #    logging.info(vocab.sentence2str(instance['span']))
    all_probs = np.stack(all_probs, axis=0)
    our_priors = np.mean(all_probs, axis=0)


    sense_keys = []
    wn_frequencies = []
    our_prob_given_wn_sense = []
    #cooccurrences = []
    for key, probs in sorted(senses.items()):
        sense_keys.append(key)
        wn_frequencies.append(len(probs))
        stacked = np.stack(probs, axis=0)
        our_prob_given_wn_sense.append(np.mean(stacked, axis=0))
        #best_senses = np.argmax(stacked, axis=1)
        #best_onehot = onehot(best_senses, stacked.shape[1])
        #cooccurrences.append(np.sum(best_onehot, axis=0))
      
    our_prob_given_wn_sense = np.stack(our_prob_given_wn_sense, axis=0)
    wn_frequencies = np.array(wn_frequencies, dtype=np.float32)
    wn_priors = wn_frequencies / np.sum(wn_frequencies)
    #cooccurrences = np.stack(cooccurrences, axis=0)

    #if options.wsd_use_frequency_counts:
    #    lemmas = get_lemmas(lemma)
    #    count = np.array([lemmas[key].count() for key in sense_keys],
    #                     dtype=np.float32) + 1.0
    #    lemma_prob = count / count.sum()
    #else:
    #    lemma_prob = np.ones([len(sense_keys)])
    
    P = np.zeros([n_senses, len(sense_keys)], dtype=np.float32)
    for i in range(n_senses):
        #P[i, :] = cooccurrences[:, i]
        #if P[i, :].sum() == 0:
        for j in range(len(sense_keys)):
            P[i, j] = our_prob_given_wn_sense[j, i] * wn_priors[j] / our_priors[i]
            #P[i, j] = lemma_prob[j] * averaged_probs[j, i]
        #P[i, :] = P[i, :] / P[i, :].sum()

    if verbose:
        logging.info('%d instances' % len(instances))
        logging.info(str(sense_keys))
        logging.info(str(lemma_prob))
        logging.info('\n' + str(averaged_probs))
        logging.info('Sense map:\n%s' % str(P))

    CACHED_MATCH_MATRICES[item] = (P, sense_keys)
    return P, sense_keys

def wordnet_sense_probs(
        instance, corpus, vocab, model, sess, options,
        use_usage_examples=False):
    #word = vocab.id2str(instance['vocab_id'])
    lemma = instance['lemma']
    pos = instance['pos']
    #n_senses = model.get_n_senses(instance['vocab_id'])
    n_senses = model.get_n_senses(vocab.str2id(lemma))
    M, keys = match_matrix(corpus, vocab, model, sess,
                           lemma, lemma, options, pos=pos)
    sense_probs = instance['sense_probs']
    sense_probs = sense_probs[:n_senses]
    wordnet_probs = np.matmul(np.expand_dims(sense_probs, axis=0), M)
    wordnet_probs = np.squeeze(wordnet_probs, axis=0)
    return wordnet_probs, M, keys

def get_definitions(w):
    lemmas = wn.lemmas(w)
    return {lemma.key(): lemma.synset().definition() for lemma in lemmas}
    
def wsd(model, vocab, sess, options, verbose=False):
    eval_corpus = WsdCorpus(options.wsd_eval_path, vocab, stem=options.lemmatize)
    eval_labels = None
    if options.wsd_eval_gold_path is not None:
        eval_labels = get_labels(options.wsd_eval_gold_path)
    all_lemmas = eval_corpus.get_all_lemmas()
    train_corpus = WsdCorpus(
            options.wsd_train_path, vocab, gold_path=options.wsd_train_gold_path,
            stem=options.lemmatize, extras=all_lemmas)
    #instances_with_word = eval_corpus.calculate_sense_probs(
    #        sess, model, options, training=False, replace_with_lemma=False)
    instances_with_lemma = eval_corpus.calculate_sense_probs(
            sess, model, options, training=False, replace_with_lemma=True,
            method=options.sense_prob_source)

    unanswerable = []
    n_correct = 0
    n_instances = 0
    for i in range(len(instances_with_lemma)):
        instance = instances_with_lemma[i]
        #word = vocab.id2str(instance['vocab_id'])
        lemma = instance['lemma']
        sense_probs_with_word, M_w, keys_w = wordnet_sense_probs(
                instance, train_corpus, vocab, model, sess, options,
                use_usage_examples=options.wsd_use_usage_examples)
        sense_probs = sense_probs_with_word
        #if lemma != word:
        #    instance_with_lemma = instances_with_lemma[i]
        #    sense_probs_with_lemma, M_l, keys_l = wordnet_sense_probs(
        #            instance_with_lemma, train_corpus, vocab, model,
        #            sess, options, use_usage_examples=use_usage_examples)
        #    if keys_w == keys_l:
        #        sense_probs = (sense_probs_with_word + sense_probs_with_lemma) / 2.0
        #    else:
        #        logging.info("Keys don't match!")

        if verbose:
            model.display_words(sess, [lemma])
            logging.info('(%s)' % ('/'.join(['%.3f' % p for p in instance['sense_probs']])))
            logging.info(vocab.sentence2str(instance['span']))
            logging.info('%s -> %s:\n%s' % (lemma, lemma, str(M_w)))
            #if lemma != word:
            #    model.display_words(sess, [lemma])
            #    logging.info('(%s)' % ('/'.join(['%.3f' % p for p in instance_with_lemma['sense_probs']])))
            #    logging.info(vocab.sentence2str(instance_with_lemma['span']))
            #    logging.info('%s -> %s:\n%s' % (lemma, lemma, str(M_l)))
            definitions = get_definitions(lemma)
            for key, prob in zip(keys_w, sense_probs):
                logging.info('p = %.4f: %s (%s)' % (prob, key, definitions[key]))

        wsd_id = instance['name']
        best_sense = np.argmax(sense_probs)
        print('%s %s' % (wsd_id, keys_w[best_sense]))

        if eval_labels is not None:
            gold = eval_labels[wsd_id]
            logging.info('Gold: %s' % str(gold))
            if len(set(gold).intersection(set(keys_w))) == 0:
                instance['sense_keys'] = gold
                unanswerable.append(instance)
            if keys_w[best_sense] in gold:
                n_correct += 1
        n_instances += 1

    if eval_labels is not None:
        logging.info('%d unanswerable instances:' % len(unanswerable))
        for instance in unanswerable:
            logging.info(100 * '*')
            logging.info('%s: %s' % (instance['lemma'], instance['pos']))
            logging.info(vocab.sentence2str(instance['span']))
            logging.info(str(instance['sense_keys']))
        percent = 100.0 * n_correct / n_instances
        logging.info('%d/%d correct (%.2f%%)' % (
                n_correct, n_instances, percent))


class WsiCorpus(object):

    def __init__(self, examples, vocab, stem=False):
        self._vocab = vocab
        self._lemmas = []
        self._pos = []
        self._instance_names = []
        self._sentences = []
        seqs = []

        for inst_name, lemma, pos, before, target, after in examples:
            before = preprocess_str(before)
            after = preprocess_str(after)
            self._instance_names.append(inst_name)
            self._lemmas.append(lemma)
            self._pos.append(pos)
            self._sentences.append(' '.join([before, target, after]))
            seqs.append(before)
            seqs.append(target)
            seqs.append(after)

        tokens, _ = tokenize(seqs, get_tokenizer())
        self._focuses = []
        self._tokens = []
        for i in range(0, len(tokens), 3):
            before_tokens, target_tokens, after_tokens = tokens[i:i + 3]
            self._focuses.append(len(before_tokens))
            self._tokens.append(before_tokens + target_tokens + after_tokens)

        if stem:
            self.stem()

    def stem(self):
        logging.info('Stemming WSI corpus')
        stemmed, alignments, tags = stem(self._tokens)
        logging.info('Finished stemming corpus')
        for i, alignment in enumerate(alignments):
            self._focuses[i] = alignment[self._focuses[i]]
            toks = stemmed[i]
            ind = self._focuses[i]
            stemmed_focus = toks[ind]
            if stemmed_focus != self._lemmas[i]:
                new_stem, suf = stem_focus(stemmed_focus, self._pos[i])
                if new_stem == self._lemmas[i]:
                    stemmed[i] = toks[:ind] + [new_stem, suf] + toks[ind+1:]
                    #logging.warn('fixed %s -> %s' % (' '.join(toks), ' '.join(stemmed[i])))
                else:
                    logging.warn('%s: stemmed focus "%s" does not match lemma "%s" in sentence "%s"' % (
                            self._instance_names[i], stemmed_focus, self._lemmas[i], ' '.join(stemmed[i])))
        self._tokens = []
        for seq in stemmed:
            self._tokens.append(seq)

    def generate_instances(self):
        for tokens, focus, lemma, pos, name, sentence in zip(
                self._tokens, self._focuses, self._lemmas,
                self._pos, self._instance_names, self._sentences):
            yield {'tokens': tokens, 'index_in_seq': focus, 'name': name,
                   'lemma': lemma, 'pos': pos, 'sentence': sentence}

    def _generate_batches(self, max_batch_size, max_seq_len):
        instance_gen = self.generate_instances()
        done = False
        while not done:
            instances = []
            while len(instances) < max_batch_size:
                try:
                    instance = next(instance_gen)
                    instances.append(instance)
                except StopIteration:
                    done = True
                    break
            yield instances, make_batch(instances, self._vocab, max_seq_len)

    def calculate_sense_probs(
            self, sess, model, max_batch_size, max_seq_length, method='prediction'):
        all_instances = []
        seen = set()
        for instances, batch in self._generate_batches(
                max_batch_size, max_seq_length):
            sense_probs = model.disambiguate(sess, batch, method=method)
            for i, instance in enumerate(instances):
                instance['sense_probs'] = sense_probs[i, :]
                all_instances.append(instance)

        return all_instances

def wsi(model, vocab, sess, options):
    path = options.wsi_path
    if options.wsi_format == 'SemEval-2010':
        path = os.path.join(path, 'test_data')
        corpus = generate_sem_eval_wsi_2010(path, vocab, stem=options.lemmatize)
        allow_multiple = False
    elif options.wsi_format == 'SemEval-2013':
        path = os.path.join(
                path,
                'contexts/senseval2-format/'
                'semeval-2013-task-13-test-data.senseval2.xml')
        corpus = generate_sem_eval_wsi_2013(path, vocab, stem=options.lemmatize)
        allow_multiple = True
    else:
        logging.error('Unrecognized WSI format: "%s"' % options.wsi_format)
        return
    instances = corpus.calculate_sense_probs(
            sess, model, options.batch_size, options.max_seq_len,
            method=options.sense_prob_source)
    lemmas = {}
    for instance in instances:
        lemmas.setdefault(instance['lemma'], []).append(instance)

    n_senses_count = np.zeros([options.max_senses_per_word + 1], dtype=np.int32)
    for lemma, instances in lemmas.items():
        for i, instance in enumerate(instances):
            lemma = instance['lemma']
            pos = instance['pos']
            sense_probs = instance['sense_probs']
            cluster_num = np.argmax(sense_probs)
            if allow_multiple:
                sense_labels = [
                        '%s.%s.%d/%.4f' % (lemma, pos, n+1, p)
                        for n, p in enumerate(sense_probs)
                        if p > options.wsi_2013_thresh or n == cluster_num]
                n_senses_count[len(sense_labels)] += 1
                score_str = ' '.join(sense_labels)
            else:
                score_str = '%s.%s.%d' % (lemma, pos, cluster_num+1)

            #logging.info(instance['sentence'])
            logging.info(to_sentence(vocab, instance['span'], instance['index_in_span']))
            logging.info('(%s)' % ('/'.join(['%.3f' % p for p in instance['sense_probs']])))
            print('%s.%s %s %s' % (lemma, pos, instance['name'], score_str))
        model.display_words(sess, [lemma])

    if allow_multiple:
        n_senses_prob = n_senses_count / np.sum(n_senses_count)
        for i, p in enumerate(n_senses_prob):
            logging.info('Probability of assigning %d senses: %.4f' % (i, p))
        mean = np.sum(np.arange(options.max_senses_per_word + 1) * n_senses_prob)
        logging.info('Mean number of senses: %.4f' % mean)

def generate_sem_eval_wsi_2013(path, vocab, stem=False):
    logging.info('reading SemEval dataset from %s' % path)

    examples = []
    tree = ElementTree.parse(path)
    root = tree.getroot()
    for lemma_element in root:
        lemma_name = lemma_element.get('item')
        logging.info(lemma_name)
        lemma = lemma_name.split('.')[0]
        pos = lemma_name.split('.')[1]

        for inst in lemma_element:
            inst_name = inst.get('id')
            context = inst[0]

            before = ''
            if context.text is not None:
                before = context.text.strip()

            target = context[0].text.strip()
            if len(target.split()) != 1:
                logging.warn(target)
                sys.exit()

            after = ''
            if context[0].tail is not None:
                after = context[0].tail.strip()

            examples.append((inst_name, lemma, pos, before, target, after))

    return WsiCorpus(examples, vocab, stem=stem)

def generate_sem_eval_wsi_2010(dir_path, vocab, stem=False):
    logging.info('reading SemEval dataset from %s' % dir_path)

    extra_mapping = {'lay': 'lie', 'figger': 'figure', 'figgered': 'figure', 'lah': 'lie',
                     'half-straightened': 'straighten'}

    examples = []
    for root_dir, dirs, files in os.walk(dir_path):  # "../paper-menuscript/resources/SemEval-2010/test_data/"):
        logging.info('In %s' % root_dir)
        #     path = root.split(os.sep)
        for file in files:
            if '.xml' in file:
                tree = ElementTree.parse(os.path.join(root_dir, file))
                root = tree.getroot()
                for child in root:
                    inst_name = child.tag
                    lemma = inst_name.split('.')[0]
                    pos = inst_name.split('.')[1]

                    #stemmed_lemma = basic_stem(lemma)

                    # pres_sent = child.text
                    #target_sent = child[0].text
                    before = ''
                    if child.text is not None:
                        before = child.text.strip()
                    
                    target_tokens = child[0].text.strip().split()
                    focus_index = -1
                    for i, token in enumerate(target_tokens):
                        token = token.lower()
                        token_lemma = lemmatize(token, pos)
                        if token_lemma == lemma or extra_mapping.get(token, '') == lemma:
                            if focus_index != -1:
                                #logging.warn('Duplicate occurrence of lemma "%s" in sentence "%s"' % (
                                #        lemma, child[0].text))
                                pass
                            else:
                                focus_index = i

                    if focus_index == -1:
                        logging.warn('Found no occurrences of lemma "%s" in sentence "%s"' % (
                                lemma, child[0].text))
                        continue

                    before += ' '.join(target_tokens[:focus_index])
                    target = target_tokens[focus_index]

                    after = ' '.join(target_tokens[focus_index+1:])
                    if child[0].tail is not None:
                        after += child[0].tail.strip()

                    #logging.info(' '.join(seq))
                    #logging.info(str(focus_index))

                    #corpus.add_example(seq, focus_index, lemma, pos, inst_name)
                    examples.append((inst_name, lemma, pos, before, target, after))

    return WsiCorpus(examples, vocab, stem=stem)


def print_embeddings(model, sess, words, vocab, options):
    embeddings = model.get_embeddings(sess)
    embedding_file = open(options.tsne_embeddings_path, 'w')
    label_file = open(options.tsne_labels_path, 'w')
    for word in words:
        vocab_id = vocab.str2id(word)
        word = vocab.id2str(vocab_id)
        model.display_words(sess, [word])
        n_senses = model.get_n_senses(vocab_id)
        for i in range(n_senses):
            embedding_str = ' '.join([
                    '%.5f' % x for x in embeddings[vocab_id, i, :]])
            embedding_file.write(embedding_str + '\n')
            label_file.write('%s_%d\n' % (word, i + 1))
    embedding_file.close()
    label_file.close()
