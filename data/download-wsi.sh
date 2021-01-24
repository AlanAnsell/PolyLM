#!/bin/bash

# Adapted from https://github.com/asafamr/bertwsi/blob/master/download_resources.sh
mkdir -p wsi
echo downloading SemEval 2013 task 13 data and evaluation code...
wget https://www.cs.york.ac.uk/semeval-2013/task13/data/uploads/semeval-2013-task-13-test-data.zip -P wsi
unzip -d wsi wsi/semeval-2013-task-13-test-data.zip
mv wsi/SemEval-2013-Task-13-test-data wsi/SemEval-2013
rm wsi/semeval-2013-task-13-test-data.zip

echo downloading SemEval 2010 task 14 data and evaluation code...
wget https://www.cs.york.ac.uk/semeval2010_WSI/files/evaluation.zip  -O wsi/se2010eval.zip
wget https://www.cs.york.ac.uk/semeval2010_WSI/files/test_data.tar.gz  -O wsi/se2010test_data.tar.gz
mkdir -p wsi/SemEval-2010
unzip -d wsi/SemEval-2010 wsi/se2010eval.zip
tar -C wsi/SemEval-2010 -xzf wsi/se2010test_data.tar.gz
rm wsi/se2010eval.zip wsi/se2010test_data.tar.gz

