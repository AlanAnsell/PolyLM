#!/bin/bash

gpus=0
lemmatize=true
sense_prob_source="prediction"
checkpoint_version=-1
wsi_2013_thresh="0.2"

train_dir=$1
shift
format=$1
shift

while [ $# -gt 0 ]; do

    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi

    shift
done

python wsi.py --flagfile=$train_dir/flags --batch_size=16 --gpus=$gpus --wsi_format=$format --lemmatize=$lemmatize --sense_prob_source=$sense_prob_source --checkpoint_version=$checkpoint_version --wsi_2013_thresh=$wsi_2013_thresh
