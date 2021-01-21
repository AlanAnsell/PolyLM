#!/bin/bash

model_path="wic.joblib"
output_dir="."
use_contextualized_reps=true
use_sense_probs=true
gpus=0
lemmatize=false
wic_path="/home/aa381/nlp/datasets/word-in-context"
mode="train"
checkpoint_version=-1
sense_prob_source="prediction"
classification_threshold=-1.0

model_dir=$1
shift

while [ $# -gt 0 ]; do

    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $param=$2
    fi

    shift
done

if [ "$mode" = "train" ]; then
    wic_gold_path=$wic_path"/all.gold.txt"
    wic_data_path=$wic_path"/all.data.txt"
    wic_train=true
else
    wic_gold_path=$wic_path"/test/test.gold.txt"
    wic_data_path=$wic_path"/test/test.data.txt"
    wic_train=false
fi

python wic.py --flagfile=$model_dir/flags --wic_gold_path=$wic_gold_path --wic_data_path=$wic_data_path --batch_size=32 --wic_model_path=$model_path --gpus=$gpus --lemmatize=$lemmatize --wic_use_contextualized_reps=$use_contextualized_reps --wic_use_sense_probs=$use_sense_probs --wic_train=$wic_train --wic_output_path=$output_dir/output.txt --checkpoint_version=$checkpoint_version --sense_prob_source=$sense_prob_source --wic_classification_threshold=$classification_threshold

if [ "$mode" = "test" ]; then
    zip $output_dir/output.zip $output_dir/output.txt
fi
