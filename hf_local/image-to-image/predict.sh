#! /bin/bash

source activate tcpy 

export "HF_HOME"="./model"

python predict.py -n black-forest-labs/FLUX.1-Fill-dev

