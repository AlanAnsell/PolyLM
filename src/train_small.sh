#!/bin/bash

LOGDIR=/research/alan/experiments/polylm/matrix
mkdir -p $LOGDIR
SRCDIR=~aa381/nlp/PolyLM/src

(
   trap "" SIGHUP 

   cd $SRCDIR

   source activate nlp4

   python train.py --train_dir=$LOGDIR --corpus_path=/research/alan/corpora/wb2019/wb2019_stemmed.txt --vocab_path=/research/alan/corpora/wb2019/stemmed_vocab.txt --extra_multisense_vocab=extra_multisense.txt --test_words="<MASK> <PAD> <S> </S> <PL> <PAST> the right head see touch cook apple bar fox cut 42" --batch_size=32  --max_senses_per_word=8 --gpus="0" --print_every=100 --test_every=3000 --dl_r=1.5 --dl_warmup_steps=2000000 --ml_warmup_steps=1000000 --learning_rate=0.00003 --n_batches=6000000 --embedding_size=128 --bert_intermediate_size=512 --n_attention_heads=8 --n_disambiguation_layers=4 --n_prediction_layers=8

   )  </dev/null >>$LOGDIR/log.out 2>>$LOGDIR/log.err &
