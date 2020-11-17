#!/bin/bash

use_training_data=true
use_definitions=true
use_usage_examples=true
use_frequency_counts=true
gpus=0
lemmatize=false
max_examples_per_sense=1000000
ignore_semeval_07=false
checkpoint_version=-1
sense_prob_source="prediction"

train_dir=$1
shift

while [ $# -gt 0 ]; do

    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $param=$2
    fi

    shift
done

python wsd.py --flagfile=$train_dir/flags  --batch_size=32 --gpus=$gpus --wsd_train_path=/research/alan/datasets/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml --wsd_train_gold_path=/research/alan/datasets/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt --wsd_eval_path=/research/alan/datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml --wsd_eval_gold_path=/research/alan/datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.key.txt --wsd_use_training_data=$use_training_data --wsd_use_definitions=$use_definitions --wsd_use_usage_examples=$use_usage_examples --wsd_use_frequency_counts=$use_frequency_counts --lemmatize=$lemmatize --wsd_max_examples_per_sense=$max_examples_per_sense --wsd_ignore_semeval_07=$ignore_semeval_07 --checkpoint_version=$checkpoint_version --sense_prob_source=$sense_prob_source
