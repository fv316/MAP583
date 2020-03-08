#!/bin/bash
set -e

# TODO kill on pors

# Available on ports 6006, 6007, 8265
tensorboard --logdir ./ecg_data/ecg/runs --port 6006 &
tense_first=$!
tensorboard --logdir ~/ray_results       --port 6007 &
tesne_second=$!

python tuning.py

kill ${tense_first}
kill ${tense_second}