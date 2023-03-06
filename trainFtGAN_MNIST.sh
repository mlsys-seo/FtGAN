#!/usr/bin/env bash

while getopts l:c: flag
do
   case "${flag}" in
      l) parameterA=${OPTARG} ;;
      c) parameterB=${OPTARG} ;;
   esac
done

python main.py \
--experiment_method=train \
--shortcut_layers 0 \
--inject_layers 0 \
--experiment_name trainFTGAN_MNIST \
--data_name mnist \
--epoch 25 \
--classifierSelectedLayer $parameterA \
--classifierSelectedChannel $parameterB \
--n_sample 128 \
--batch_size 128 \
--enc_dim 4 \
--dec_dim 4 \
--dis_dim 4
