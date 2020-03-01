#!/bin/bash

set -e

function test_model() {
    arch=$1
    model_name=$2

    python commander.py \
        --dataset ecg   \
        --name ecg_${arch}_${model_name}_optsgd_lr1e-2_lrStep0.5_bsz128 \
        --num-classes 5 \
        --epochs 5 \
        --root-dir ecg_data \
        --arch ${arch} \
        --model-name ${model_name} \
        --batch-size 128 \
        --short-run
}

test_model cnn1d cnn1d_3
test_model resnet1d resnet1d_18
test_model resnet1d resnet1d_10
test_model resnet1d_v2 resnet1d_v2_18
test_model resnet1d_v2 resnet1d_v2_10


