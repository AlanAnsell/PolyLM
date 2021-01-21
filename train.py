import os
import sys
import shutil
import time

import tensorflow as tf

import init
from data import Corpus
import polylm


def publish_source(src_dir):
    should_publish = lambda s: s.endswith('.py') or s.endswith('.sh')
    my_dir = sys.path[0]
    to_publish = [f for f in os.listdir(my_dir) if os.path.isfile(f) and should_publish(f)]
    for f in to_publish:
        shutil.copyfile(os.path.join(my_dir, f), os.path.join(src_dir, f))

def main(unused_argv):
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    options, vocab, multisense_vocab, tf_config = init.init()
    model = polylm.PolyLM(
            vocab, options, multisense_vocab=multisense_vocab, training=True)
    test_words = options.test_words.split()

    if not os.path.exists(options.model_dir):
        os.makedirs(options.model_dir)
    
    src_dir = os.path.join(options.model_dir, 'src_%d' % int(time.time()))
    os.makedirs(src_dir)
    publish_source(src_dir)

    flags_str = options.flags_into_string()
    with open(os.path.join(options.model_dir, 'flags'), 'w') as f:
        f.write(flags_str)

    corpus = Corpus(options.corpus_path, vocab)
    with tf.Session(config=tf_config) as sess:
        model.attempt_restore(sess, options.model_dir, False)
        model.train(corpus, sess, test_words=test_words) 


if __name__ == "__main__":
    tf.app.run()
