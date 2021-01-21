#!/bin/bash

LOGDIR=/research/alan/experiments/polylm/base
mkdir -p $LOGDIR
SRCDIR=/home/aa381/nlp/PolyLM/src

(
   trap "" SIGHUP 

   cd $SRCDIR

   source activate nlp5

   python train.py --model_dir=$LOGDIR --corpus_path=/research/alan/corpora/wb2019/wb2019_stemmed.txt --vocab_path=/research/alan/corpora/wb2019/stemmed_vocab.txt --n_senses_file=/research/alan/corpora/wb2019/n_senses.txt --test_words="<MASK> <PAD> a right see touch cook apple bar fox cut rock bank" --batch_size=32  --max_senses_per_word=8 --gpus="0" --print_every=100 --test_every=3000 --dl_r=1.5 --dl_warmup_steps=2000000 --ml_warmup_steps=1000000 --learning_rate=0.00003 --n_batches=6000000 --embedding_size=256 --bert_intermediate_size=1024 --n_attention_heads=8 --n_disambiguation_layers=4 --n_prediction_layers=12 -min_occurrences_for_vocab=500 --masking_policy="0.8 0.1 0.1"

   )  </dev/null >>$LOGDIR/log.out 2>>$LOGDIR/log.err &
