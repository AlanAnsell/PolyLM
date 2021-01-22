# PolyLM - Neural Word Sense Model

## Setup
Recommended environment is Python >= 3.7 and TensorFlow 1.15, although earlier versions of TensorFlow 1 may also work.

    git clone https://github.com/AlanAnsell/PolyLM.git
    cd PolyLM
    conda create -n polylm python=3.7 tensorflow-gpu=1.15
    conda activate polylm
    pip install -r requirements.txt

Currently [Stanford CoreNLP's part-of-speech tagger](https://nlp.stanford.edu/software/tagger.shtml#Download) is also required to perform lemmatization when evaluating on WSI.



