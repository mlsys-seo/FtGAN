#!/usr/bin/env bash

python main.py \
--experiment_method=test \
--shortcut_layers 0 \
--inject_layers 0 \
--experiment_name FtGAN_mnist_l4_ch234 \
--data_name mnist \
--classifierSelectedChannel 234 \
--classifierSelectedLayer 4 \
--n_sample 100 \
--batch_size 100 \
--enc_dim 4 \
--dec_dim 4 \
--dis_dim 4
