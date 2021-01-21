#!/bin/bash

gpus=0
lemmatize=true
sense_prob_source="prediction"
checkpoint_version=-1
wsi_2013_thresh="0.2"
pos_tagger_root=""

wsi_dir=$1
shift
format=$1
shift
model_dir=$1
shift

while [ $# -gt 0 ]; do

    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi

    shift
done

OUTFILE=/tmp/wsi-predictions.txt

python wsi.py --flagfile=$model_dir/flags --model_dir=$model_dir --batch_size=16 --gpus=$gpus --wsi_path=$wsi_dir --wsi_format=$format --lemmatize=$lemmatize --sense_prob_source=$sense_prob_source --checkpoint_version=$checkpoint_version --wsi_2013_thresh=$wsi_2013_thresh --pos_tagger_root=$pos_tagger_root > $OUTFILE

if [[ $format == "SemEval-2010" ]]; then
    java -jar wsi-eval/fscore.jar $OUTFILE $wsi_dir/evaluation/unsup_eval/keys/all.key all
    java -jar wsi-eval/vmeasure.jar $OUTFILE $wsi_dir/evaluation/unsup_eval/keys/all.key all
fi

if [[ $format == "SemEval-2013" ]]; then
    java -jar wsi-eval/fuzzy-bcubed.jar $wsi_dir/keys/gold/all.key $OUTFILE
    java -jar wsi-eval/fuzzy-nmi.jar $wsi_dir/keys/gold/all.key $OUTFILE
fi


