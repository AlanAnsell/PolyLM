# PolyLM - Neural Word Sense Model

## Setup
Recommended environment is Python >= 3.7 and TensorFlow 1.15, although earlier versions of TensorFlow 1 may also work.

    git clone https://github.com/AlanAnsell/PolyLM.git
    cd PolyLM
    conda create -n polylm python=3.7 tensorflow-gpu=1.15
    conda activate polylm
    pip install -r requirements.txt

Currently [Stanford CoreNLP's part-of-speech tagger](https://nlp.stanford.edu/software/tagger.shtml#Download) is also required to perform lemmatization when evaluating on WSI.


## Training
The training corpus should be stored in a text file, where each line is a training example. The maximum length of a line should be the maximum sequence length minus two (to allow for the addition of beginning and end of sequence tokens).

The first step is to create a vocabulary file for the corpus:

    from data import Vocabulary
    vocab = Vocabulary('/path/to/corpus/corpus.txt', min_occurrences=500, build=True)
    vocab.write_vocab_file('/path/to/corpus/vocab.txt')

The `min_occurrences` parameter controls which tokens are included in the vocabulary. Tokens which appear fewer than `min_occurrences` times in the corpus will be replaced with an `<UNK>` token during training.

The following command can be used to train a model with the same parameters as PolyLM_{BASE}:

    python train.py --model_dir=/where/to/save/model
                    --corpus_path=/path/to/corpus/corpus.txt
                    --vocab_path=/path/to/corpus/vocab.txt
                    --embedding_size=256
                    --bert_intermediate_size=1024
                    --n_disambiguation_layers=4
                    --n_prediction_layers=12
                    --max_senses_per_word=8
                    --min_occurrences_for_vocab=500
                    --min_occurrences_for_polysemy=20000
                    --max_seq_len=128
                    --gpus=0
                    --batch_size=32
                    --n_batches=6000000
                    --dl_warmup_steps=2000000
                    --ml_warmup_steps=1000000
                    --dl_r=1.5
                    --ml_coeff=0.1
                    --learning_rate=0.00003
                    --print_every=100
                    --save_every=10000

Notes:
 * Tokens appearing at least `max_occurrences_for_polysemy` times in the corpus will be allocated `max_senses_per_word` senses; all other tokens will be allocated a single sense. If you wish to allocate a custom number of senses per token, you may createa file where each line is of the form `<token> <n_senses>` and pass it as the value of command line parameter `--n_senses_file`. Note that the `--max_senses_per_word` parameter must still be supplied.
 * To train on multiple GPUs, pass them as a comma-separated string, e.g. `--gpus=0,1`. Note that `--batch_size` is per-GPU, so to train with a total batch size of 32 on 2 GPUs, you should set `--batch_size=16`.
 * You may optionally supply a `--test_words` argument to specify a number of words for which to print nearest-neighbour senses during training. For instance, setting `--test_words="bank, rock, bar"` and `--test_every=1000` would cause nearest-neighbour senses for all senses of words "bank", "rock" and "bar" to be printed each 1,000 batches. This can be useful for ensuring that training is proceeding as intended.


