#! /bin/bash

source activate hfpy

python predict.py -n black-forest-labs/FLUX.1-dev  -p hf-inference -s 30

