#!/bin/bash

LOGDIR=/mnt/hdd/aja63/experiments/polylm-multi/small-1
mkdir -p $LOGDIR
SRCDIR=/home/aja63/projects/polylm-multi/PolyLM/src

(
   trap "" SIGHUP 

   cd $SRCDIR

   source activate polylm

   python train.py --train_dir=$LOGDIR --corpus_path=/mnt/hdd/aja63/datasets/wiki/XLM/corpus.txt --vocab_path=/mnt/hdd/aja63/datasets/wiki/XLM/vocab.txt --n_senses_file=/mnt/hdd/aja63/datasets/wiki/XLM/n_senses.txt --test_words="<MASK> <PAD> a hay sierra man gross right see touch cook apple bar fox cut" --batch_size=16  --max_senses_per_word=8 --gpus="0,1" --print_every=100 --test_every=3000 --dl_r=1.5 --dl_warmup_steps=2000000 --ml_warmup_steps=1000000 --learning_rate=0.00003 --n_batches=6000000 --embedding_size=128 --bert_intermediate_size=512 --n_attention_heads=8 --n_disambiguation_layers=4 --n_prediction_layers=8 --pos_tagger_root=/home/aja63/tools/stanford-postagger-2018-10-16 --min_occurrences_for_vocab=1 --masking_policy="0.9 0.0 0.1"

   )  </dev/null >>$LOGDIR/log.out 2>>$LOGDIR/log.err &
