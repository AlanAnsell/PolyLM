import tensorflow as tf

import init
import polylm
import util


def main(unused_argv):
    options, vocab, multisense_vocab, tf_config = init.init()
    model = polylm.PolyLM(
            vocab, options, multisense_vocab=multisense_vocab, training=False)

    with tf.Session(config=tf_config) as sess:
        model.attempt_restore(sess, options.train_dir, True)
        util.wic(model, vocab, sess, options)


if __name__ == "__main__":
    tf.app.run()
