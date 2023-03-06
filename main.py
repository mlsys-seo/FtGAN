from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime

import data
import global_args

import train_test_mnist

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()

att_default = ['']

parser.add_argument('--atts', dest='atts', default=att_default, nargs='+', help='attributes to learn')
parser.add_argument('--img_size', dest='img_size', type=int, default=128)
parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)

# training
parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
parser.add_argument('--n_sample', dest='n_sample', type=int, default=64, help='# of sample images')

# others
parser.add_argument('--use_cropped_img', dest='use_cropped_img', action='store_true')
parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
parser.add_argument('--data_name', dest='data_name', default="Celeba")
parser.add_argument('--hyper_param', dest='hyper_param', type=float, default=100.0)
parser.add_argument('--hyper_param2', dest='hyper_param2', type=float, default=10.0)
parser.add_argument('--description', dest='description', default="description")
parser.add_argument('--classifierSelectedChannel', dest='classifierSelectedChannel', type=int, default=0)
parser.add_argument('--classifierSelectedLayer', dest='classifierSelectedLayer',  type=int, default=1)

parser.add_argument('--trainRandomNoiseRange', dest='trainRandomNoiseRange', type=int, default=10)
parser.add_argument('--testRandomNoiseRange', dest='testRandomNoiseRange', type=int, default=100)
parser.add_argument('--lossLambda', dest='lossLambda', type=float, default=0.1)
parser.add_argument('--trainClassification', dest='trainClassification', type=bool, default=False)
parser.add_argument('--experiment_method', dest='experiment_method', default='train')

parser.add_argument('--output_path', dest='output_path', default='./output/')

args = parser.parse_args()

global_arg = global_args.global_args(args)
global_arg.n_att = 1

if global_arg.experiment_method == "train":
    train_test_mnist.train(global_arg)

elif global_arg.experiment_method == "test":
    if global_arg.data_name == "mnist":
        train_test_mnist.test(global_arg)
