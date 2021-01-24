# PolyLM - Neural Word Sense Model

## Setup
Recommended environment is Python >= 3.7 and TensorFlow 1.15, although earlier versions of TensorFlow 1 may also work.

    git clone https://github.com/AlanAnsell/PolyLM.git
    cd PolyLM
    conda create -n polylm python=3.7 tensorflow-gpu=1.15
    conda activate polylm
    pip install -r requirements.txt

Currently [Stanford CoreNLP's part-of-speech tagger v3.9.2](https://nlp.stanford.edu/software/stanford-postagger-2018-10-16.zip) is also required to perform lemmatization when evaluating on WSI.


## Pretrained Models
The code has been rewritten to achieve much better performance since the paper was published and the old checkpoints are no longer compatible. However this has enabled us to train a larger version of the model, PolyLM<sub>LARGE</sub>, which achieves state-of-the-art performance on both WSI datasets used for evaluation, and is available below. We are also training versions of the SMALL and BASE models which are compatible with the new code.

|Model|Embedding size|Num. params|FScore|VMeasure|SemEval-2010 AVG|FBC|FNMI|SemEval-2013 AVG|
|-----|--------------|-----------|------|--------|----------------|---|----|----------------|
|[PolyLM<sub>LARGE</sub>](https://docs.google.com/uc?export=download&id=1HON62UBIsEiwLTPHkdFrCau47tIfz948)|384|90M|67.5|**43.6**|**54.3**|**66.7**|**23.7**|**39.7**|
|PolyLM<sub>BASE</sub>|256|54M|65.8|40.5|51.6|64.8|23.0|38.6|
|PolyLM<sub>SMALL</sub>|128|24M|65.6|35.7|48.4|64.5|18.5|34.5|
|Amrami and Goldberg (2019) - BERT<sub>LARGE</sub>|1024|340M|**71.3**|40.4|53.6|64.0|21.4|37.0|

It may be most convenient to use the download scripts provided in the `models` folder, e.g. to download PolyLM<sub>LARGE</sub>, do

    cd models
    ./download-lemmatized-large.sh

## Training
The training corpus should be stored in a text file, where each line is a training example. The maximum length of a line should be the maximum sequence length minus two (to allow for the addition of beginning and end of sequence tokens).

The first step is to create a vocabulary file for the corpus:

    from data import Vocabulary
    vocab = Vocabulary('/path/to/corpus/corpus.txt', min_occurrences=500, build=True)
    vocab.write_vocab_file('/path/to/corpus/vocab.txt')

The `min_occurrences` parameter controls which tokens are included in the vocabulary. Tokens which appear fewer than `min_occurrences` times in the corpus will be replaced with an `<UNK>` token during training.

The following command can be used to train a model with the same parameters as PolyLM<sub>BASE</sub>:

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


## Word Sense Induction

First download the SemEval 2010 and 2013 WSI datasets:

    cd data
    ./download-wsi.sh
    cd ..

Activate NLTK's WordNet capabilities:

    python -c "import nltk; nltk.download('wordnet')"

PolyLM evaluation can be performed as follows:

    ./wsi.sh data/wsi/SemEval-2010 SemEval-2010 /path/to/model --gpus 0 --pos_tagger_root /path/to/stanford/pos/tagger
    ./wsi.sh data/wsi/SemEval-2013 SemEval-2013 /path/to/model --gpus 0 --pos_tagger_root /path/to/stanford/pos/tagger

Note that inference is only supported on a single GPU currently, but is generally very fast.
    
