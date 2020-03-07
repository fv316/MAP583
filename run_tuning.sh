#!/bin/bash

# Available on ports 6006, 6007, 8265
tensorboard --logdir ./ecg_data/ecg/runs --port 6006 &
tensorboard --logdir ~/ray_results       --port 6007 &
python tuning.py