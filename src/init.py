import os
import logging

import tensorflow as tf

from data import Vocabulary

logging.basicConfig(level=logging.INFO)

# High-level options
tf.app.flags.DEFINE_string("gpus", "0", "Which GPUs to use, e.g. '0,1'")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train")
tf.app.flags.DEFINE_integer(
        "checkpoint_version", -1,
        "Which saved model checkpoint version to load.")

# Model hyperparameters
tf.app.flags.DEFINE_integer(
        "embedding_size", 128, "Size of sense embeddings.")
tf.app.flags.DEFINE_integer(
        "bert_intermediate_size", 512, "Number of filters.")
tf.app.flags.DEFINE_integer(
        "n_disambiguation_layers", 4,
        "Number of Transformer blocks in disambiguation layer.")
tf.app.flags.DEFINE_integer(
        "n_prediction_layers", 8,
        "Number of Transformer blocks in prediction layer.")
tf.app.flags.DEFINE_integer(
        "n_attention_heads", 8, "Number of attention heads.")
tf.app.flags.DEFINE_bool(
        "use_disambiguation_layer", True,
        "Whether to have a disambiguation layer.")
tf.app.flags.DEFINE_integer(
        "max_senses_per_word", 8,
        "Maximum number of senses a token may have.")
tf.app.flags.DEFINE_float(
        "ml_coeff", 0.1, "Coefficient of match loss.")
tf.app.flags.DEFINE_float(
        "dl_r", 1.5, "Distinctness loss coefficient r.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Dropout probability.")

# Training hyperparameters
tf.app.flags.DEFINE_integer(
        "n_batches", 6000000, "Number of batches to train for.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size per GPU.")
tf.app.flags.DEFINE_integer("max_seq_len", 128, "Maximum sequence length.")
tf.app.flags.DEFINE_float(
        "max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("learning_rate", 0.00003, "Learning rate.")
tf.app.flags.DEFINE_integer(
        "lr_warmup_steps",
        10000,
        "Number of batches over which learning rate is linearly increased at "
        "start of training.")
tf.app.flags.DEFINE_bool(
        "anneal_lr", True, "Whether to linearly decay learning rate during "
        "training.")

tf.app.flags.DEFINE_integer(
        "dl_warmup_steps", 2000000,
        "Number of batches over which distinctness loss exponent r is linearly "
        "increased at beginning of training.")
tf.app.flags.DEFINE_integer(
        "ml_warmup_steps", 1000000,
        "Number of batches over which match loss coefficient is linearly "
        "increased at beginning of training.")
tf.app.flags.DEFINE_float(
        "mask_prob", 0.15, "The proportion of tokens which are masked.")
tf.app.flags.DEFINE_string(
        "masking_policy", "0.8 0.1 0.1",
        "Policy for masking input tokens, consisting of three space-separated "
        "probabilities, the first denoting the proportion of target tokens "
        "replaced with [MASK], the second denoting the proportion left "
        "unchanged, and the third denoting the proportion replaced with a "
        "random token.")

# Preprocessing parameters
tf.app.flags.DEFINE_integer(
        "min_occurrences_for_vocab", 500,
        "Minimum number of times a token needs to occur in the corpus to be "
        "included in the vocabulary.")
tf.app.flags.DEFINE_integer(
        "min_occurrences_for_polysemy", 20000,
        "Minimum number of times a token needs to occur in the corpus to be "
        "considered for having multiple senses.")
tf.app.flags.DEFINE_bool(
        "lemmatize", True, "Whether to lemmatize during preprocessing.")

# Display parameters
tf.app.flags.DEFINE_string(
        "test_words", "",
        "Words to display neighbouring embeddings for during training.")
tf.app.flags.DEFINE_integer(
        "print_every", 100, "How many batches per stats print-out.")
tf.app.flags.DEFINE_integer(
        "save_every", 10000, "How many batches per save.")
tf.app.flags.DEFINE_integer(
        "test_every", 500, "How many batches per display of test words.")
tf.app.flags.DEFINE_integer(
        "keep", 1, "How many checkpoints to keep.")


# File and directory parameters
tf.app.flags.DEFINE_string(
        "extra_multisense_vocab", "",
        "File containing multisense tokens - overrides "
        "min_occurrences_for_polysemy.")
tf.app.flags.DEFINE_string(
        "train_dir", "",
        "Training directory to save the model parameters and other info.")
tf.app.flags.DEFINE_string(
        "corpus_path", "", "Text file containing corpus.")
tf.app.flags.DEFINE_string(
        "vocab_path", "", "Vocab file.")
tf.app.flags.DEFINE_string(
        "pos_tagger_root", "", "Root directory of Stanford POS Tagger.")

# Evaluation parameters
tf.app.flags.DEFINE_string("sense_prob_source", "prediction", "")

# WSI
tf.app.flags.DEFINE_string("wsi_path", "", "")
tf.app.flags.DEFINE_string("wsi_format", "", "")
tf.app.flags.DEFINE_float("wsi_2013_thresh", 0.2, "")

# WIC
tf.app.flags.DEFINE_bool("wic_train", True, "")
tf.app.flags.DEFINE_string("wic_data_path", "", "")
tf.app.flags.DEFINE_string("wic_gold_path", "", "")
tf.app.flags.DEFINE_bool("wic_use_contextualized_reps", True, "")
tf.app.flags.DEFINE_bool("wic_use_sense_probs", True, "")
tf.app.flags.DEFINE_float("wic_classification_threshold", -1.0, "")
tf.app.flags.DEFINE_string("wic_model_path", "", "")
tf.app.flags.DEFINE_string("wic_output_path", "", "")

# WSD
tf.app.flags.DEFINE_string("wsd_train_path", "", ".")
tf.app.flags.DEFINE_string("wsd_eval_path", "", ".")
tf.app.flags.DEFINE_string("wsd_train_gold_path", None, ".")
tf.app.flags.DEFINE_string("wsd_eval_gold_path", None, ".")
tf.app.flags.DEFINE_bool("wsd_use_training_data", True, ".")
tf.app.flags.DEFINE_bool("wsd_use_usage_examples", True, ".")
tf.app.flags.DEFINE_bool("wsd_use_definitions", True, ".")
tf.app.flags.DEFINE_bool("wsd_use_frequency_counts", True, ".")
tf.app.flags.DEFINE_bool("wsd_ignore_semeval_07", True, ".")
tf.app.flags.DEFINE_integer("wsd_max_examples_per_sense", 1000000, ".")


def get_multisense_vocab(path, vocab, FLAGS):
    multisense_vocab = set(
            [i for i in range(vocab.size)
             if vocab.get_n_occurrences(i) >=
                    FLAGS.min_occurrences_for_polysemy])
    logging.info('Base multisense vocab: %d tokens occur more than %d times' % (
            len(multisense_vocab), FLAGS.min_occurrences_for_polysemy))
    if path:
        extra_multisense_vocab = set()
        with open(path, 'r') as f:
            for line in f:
                vocab_id = vocab.str2id(line.strip())
                if vocab_id == vocab.unk_vocab_id:
                    logging.warn('Token "%s" not in vocabulary.' % line.strip())
                extra_multisense_vocab.add(vocab_id)
        logging.info('Extra multisense vocab: %d tokens' % len(extra_multisense_vocab))
        multisense_vocab = multisense_vocab | extra_multisense_vocab
        logging.info('Deduplicated multisense vocab: %d tokens' % len(multisense_vocab))
    return sorted(list(multisense_vocab))


def init():
    options = tf.app.flags.FLAGS
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus

    if not options.train_dir or not options.vocab_path:
        raise Exception('You need to specify --train_dir and --vocab_path')

    vocab = Vocabulary(options.vocab_path, options)
    multisense_vocab = get_multisense_vocab(
            options.extra_multisense_vocab, vocab, options)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    tf_config.allow_soft_placement = True

    return options, vocab, multisense_vocab, tf_config

