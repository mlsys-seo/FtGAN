from functools import partial

import tensorflow as tf
import numpy as np
import traceback

import data
import models
import pylib
import tflib as tl

from tqdm import tqdm

from alexnet import *
from datamodel_config import dataModelConfig
from datamodel_channel_config import dataModelChannel
import loss_function

from train_utils import *
from check_channel import *


def load_dataset(global_args, sess):
    tr_data = data.Mnist('./data',
                         global_args.atts,
                         global_args.img_size,
                         global_args.batch_size,
                         part='train',
                         sess=sess,
                         shuffle=False,
                         crop=not global_args.use_cropped_img,
                         experiment_name=global_args.experiment_name)

    if global_args.experiment_method == 'train':
        val_data_classifier = None
        val_data = data.Mnist('./data',
                              global_args.atts,
                              global_args.img_size,
                              global_args.n_sample,
                              part='val',
                              shuffle=False,
                              sess=sess,
                              crop=not global_args.use_cropped_img,
                              experiment_name=global_args.experiment_name)
    else:
        val_data_classifier = data.Mnist('./data',
                                         global_args.atts,
                                         global_args.img_size,
                                         global_args.n_sample,
                                         part='val',
                                         shuffle=False,
                                         sess=sess,
                                         crop=not global_args.use_cropped_img,
                                         experiment_name=global_args.experiment_name)
        val_data = data.Mnist('./data',
                              global_args.atts,
                              global_args.img_size,
                              global_args.n_sample,
                              part='val',
                              shuffle=False,
                              sess=sess,
                              crop=not global_args.use_cropped_img,
                              experiment_name=global_args.experiment_name)

    return tr_data, val_data, val_data_classifier


def configure_targetmodel(global_args, sess):
    inputSize = 32
    dataLabel = 10
    n_att = global_args.n_att

    datacfg = dataModelConfig(inputSize, global_args.classifierSelectedChannel, global_args.classifierSelectedLayer, dataLabel, n_att)
    datacfg.SetClassifierSelectedChannelList(global_args.classifierSelectedChannel)
    datacfg.LoadAlexStructure(global_args.data_name)

    if not global_args.trainClassification:
        datacfg.LoadAlexWeight(sess, global_args.data_name, global_args.experiment_name)

    datacfg.resize = global_args.img_size

    return datacfg


def set_model_gan(global_args, datacfg, colorChannel=3):
    Genc = partial(models.Genc, dim=global_args.enc_dim, n_layers=global_args.enc_layers)
    Gdec = partial(models.Gdec, dim=global_args.dec_dim, n_layers=global_args.dec_layers, shortcut_layers=global_args.shortcut_layers, inject_layers=global_args.inject_layers, colorChannel=colorChannel)
    D_info = partial(models.D_info, n_att=global_args.n_att, dim=global_args.dis_dim, fc_dim=global_args.dis_fc_dim, n_layers=global_args.dis_layers)

    return Genc, Gdec, D_info


def set_noise_gan_absolute(global_args, datacfg, data_channel, image, noiseSet=False):
    data_channel.alexInputTensor = image
    data_channel.resizeInputTensor(datacfg.resize, datacfg.inputSize, datacfg.inputSize)
    data_channel.extractChannelOutput(datacfg.alex_keep_prob, datacfg.alexNumLabel, datacfg.classifierSelectedLayer, global_args.experiment_name)
    data_channel.extractSelectedChannel(datacfg.classifierSelectedChannelList)

    if noiseSet:
        data_channel.makeRandomNoiseAddInput(datacfg.classifierSelectedChannelList, -0.67, 2)

    return data_channel


def set_loss_gan_contchintensity(global_args, datacfg,
                                 Genc, Gdec, D_info,
                                 xa, _a, _b, lr,
                                 intensityLossLamda,
                                 reconstructionLossRatio,
                                 contLossLambda,
                                 data_channel_ori):
    z = Genc(xa)
    xb_ = Gdec(z, _b)
    with tf.control_dependencies([xb_]):
        xa_ = Gdec(z, _a)

    xa_logit_gan, _, _, _ = D_info(xa)
    xb__logit_gan, _, xb__r_cont_mu, xb__r_cont_var = D_info(xb_)

    data_channel_gen = dataModelChannel()
    set_noise_gan_absolute(global_args, datacfg, data_channel_gen, xb_, False)

    # cont loss
    eplison = (xb__r_cont_mu - _b) / xb__r_cont_var
    cont_loss = -tf.reduce_mean(tf.reduce_sum(-0.5 * tf.log(2 * np.pi * xb__r_cont_var + 1e-8) -
                                              0.5 * tf.square(eplison), axis=1))
    cont_loss = cont_loss * contLossLambda

    # discriminator losses
    d_loss_gan = -(tf.reduce_mean(xa_logit_gan) - tf.reduce_mean(xb__logit_gan))
    gp = models.gradient_penalty(D_info, xa, xb_)
    d_loss = loss_function.makeCustomDLoss(d_loss_gan, gp) + cont_loss

    # generator losses
    xb__loss_gan = -tf.reduce_mean(xb__logit_gan)
    xa__loss_rec = tf.losses.absolute_difference(xa, xa_) * reconstructionLossRatio
    xb__loss_att_classifier = loss_function.make_xbLossAttClassifier(data_channel_gen.chListTensor,
                                                                     data_channel_ori.randomNoiseAddInputTensorList[0],
                                                                     datacfg.classifierSelectedChannelList[0])
    xb__loss_att_classifier = xb__loss_att_classifier * intensityLossLamda
    g_loss = xb__loss_gan + xa__loss_rec + xb__loss_att_classifier + cont_loss

    # optimizer
    d_var = tl.trainable_variables('D')
    d_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss, var_list=d_var)

    g_var = tl.trainable_variables('G')
    g_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)

    # summary
    d_summary = tl.summary({
        d_loss_gan: 'd_loss_gan',
        gp: 'gp',
        xa_logit_gan: 'xa_logit_gan',
        xb__logit_gan: 'xb__logit_gan',
        cont_loss: 'cont_loss',
        d_loss: 'd_loss',
    }, scope='D')

    lr_summary = tl.summary({lr: 'lr'}, scope='Learning_Rate')
    d_summary = tf.summary.merge([d_summary, lr_summary])

    g_summary = tl.summary({
        xb__loss_gan: 'xb__loss_gan',
        xa__loss_rec: 'xa__loss_rec',
        cont_loss: 'cont_loss',
        xb__loss_att_classifier: 'xb__loss_att_classifier',
        g_loss: 'g_loss',
    }, scope='G')

    return d_step, g_step, d_summary, g_summary


def train(global_args):
    sess = tl.session()

    tr_data, val_data, val_data_classifier = load_dataset(global_args, sess)
    datacfg = configure_targetmodel(global_args, sess)

    colorChannel = 1
    Genc, Gdec, D_info = set_model_gan(global_args, datacfg, colorChannel)

    # ================  set placeholder ================== #
    # lr        = learning rate
    # xa        = real image
    # xa_sample = image placeholder
    # b_sample  = channel intensity

    lr = tf.placeholder(dtype=tf.float32, shape=[])
    xa = tr_data.batch_op[0]

    xa_sample = tf.placeholder(tf.float32, shape=[None, global_args.img_size, global_args.img_size, colorChannel])
    _b_sample = tf.placeholder(tf.float32, shape=[None, global_args.n_att])
    # ==================================================== #

    # original channel Intensity
    data_channel_ori = dataModelChannel()
    set_noise_gan_absolute(global_args, datacfg, data_channel_ori, xa, True)

    data_channel_test = dataModelChannel()
    set_noise_gan_absolute(global_args, datacfg, data_channel_test, xa_sample, False)

    max_intensity = max_chIntensity_digits(global_args, sess, datacfg, tr_data, datacfg.classifierSelectedLayer, datacfg.classifierSelectedChannelList[0])

    _a = tf.expand_dims(data_channel_ori.chListTensor[:, datacfg.classifierSelectedChannelList[0]], axis=1) / max_intensity
    _b = tf.expand_dims(data_channel_ori.randomNoiseAddInputTensorList[0][:, datacfg.classifierSelectedChannelList[0]], axis=1) / max_intensity

    intensityLossLamda = tf.Variable(0.1)
    reconstructionLossRatio = tf.Variable(100.0)
    contLossLambda = tf.Variable(0.01)

    d_step, g_step, d_summary, g_summary = set_loss_gan_contchintensity(global_args, datacfg,
                                                                        Genc, Gdec, D_info,
                                                                        xa, _a, _b, lr,
                                                                        intensityLossLamda,
                                                                        reconstructionLossRatio,
                                                                        contLossLambda,
                                                                        data_channel_ori)

    att_sample_input = _b_sample
    x_sample = Gdec(Genc(xa_sample, is_training=False), att_sample_input, is_training=False)

    # training ======================================================================================================= #
    it_cnt, update_cnt = tl.counter()
    saver = tf.train.Saver(max_to_keep=1)

    ckpt_dir = global_args.output_path+'/%s/checkpoints' % global_args.experiment_name
    pylib.mkdir(ckpt_dir)

    try:
        tl.load_checkpoint(ckpt_dir, sess)
    except:
        sess.run(tf.global_variables_initializer())
        datacfg.LoadAlexWeight(sess, global_args.data_name, global_args.experiment_name)

    try:
        # data for sampling
        contSize = 15
        original_images_list = list()
        original_labels_list = list()
        b_sample_ipt_list = list()

        min_range = -0.67
        max_range = 2

        for idx in range(5):
            original_images, original_labels = val_data.get_next()
            original_labels_ori = original_labels

            original_images_list.append(original_images)
            original_labels_list.append(original_labels_ori)

            # for absolute
            a_sample_ipt = sess.run(data_channel_test.chListTensor, feed_dict={xa_sample: original_images, datacfg.alex_keep_prob: 1.0}) / max_intensity
            a_sample_ipt = np.expand_dims(a_sample_ipt[:, datacfg.classifierSelectedChannelList[0]], axis=1)

            contList = np.linspace(min_range, max_range, contSize)
            b_sample_ipt = [a_sample_ipt + 0.0]

            for i in range(contSize):
                tmp = np.array(a_sample_ipt, copy=True)
                tmp = (tmp + contList[i] * tmp)
                b_sample_ipt.append(tmp)
            b_sample_ipt_list.append(b_sample_ipt)

        #train epoch start
        it_per_epoch = len(tr_data) // (global_args.batch_size * (global_args.n_d + 1))
        max_it = global_args.epoch * it_per_epoch

        for it in tqdm(range(sess.run(it_cnt), max_it)):
            with pylib.Timer(is_output=False) as t:
                sess.run(update_cnt)

                epoch = it // it_per_epoch
                it_in_epoch = it % it_per_epoch + 1
                lr_ipt = global_args.lr_base / (10 ** (epoch // 100))  # learning rate

                if epoch == 0 and it % it_per_epoch == 0:
                    sess.run(reconstructionLossRatio.assign(100.0))
                    sess.run(contLossLambda.assign(1.0))
                    sess.run(intensityLossLamda.assign(1.0))
                elif epoch == 5 and it % it_per_epoch == 0:
                    sess.run(reconstructionLossRatio.assign(50.0))
                    sess.run(contLossLambda.assign(2.0))
                    sess.run(intensityLossLamda.assign(1.0))
                elif epoch == 10 and it % it_per_epoch == 0:
                    sess.run(reconstructionLossRatio.assign(25.0))
                    sess.run(contLossLambda.assign(3.0))
                    sess.run(intensityLossLamda.assign(1.0))

                # train D
                for i in range(global_args.n_d):
                    d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={lr: lr_ipt})

                # train G
                g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={lr: lr_ipt})

                # display
                display(epoch, it, it_in_epoch, it_per_epoch, t)

                # save
                save(epoch, it, it_in_epoch, it_per_epoch, saver, sess, ckpt_dir)
    except:
        traceback.print_exc()
    finally:
        save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, global_args.epoch, it_in_epoch, it_per_epoch))
        print('Model is saved at %s!' % save_path)
        sess.close()


def test(global_args):
    sess = tl.session()

    tr_data, val_data, val_data_classifier = load_dataset(global_args, sess)
    datacfg = configure_targetmodel(global_args, sess)

    colorChannel = 1
    Genc, Gdec, D_info = set_model_gan(global_args, datacfg, colorChannel)

    xa_sample = tf.placeholder(tf.float32, shape=[None, global_args.img_size, global_args.img_size, colorChannel])
    _b_sample = tf.placeholder(tf.float32, shape=[None, global_args.n_att])

    data_channel_test = dataModelChannel()
    set_noise_gan_absolute(global_args, datacfg, data_channel_test, xa_sample, False)

    max_intensity = max_chIntensity_digits(global_args, sess, datacfg, tr_data, datacfg.classifierSelectedLayer, datacfg.classifierSelectedChannelList[0])
    print('max_intensity', max_intensity)

    # original
    att_sample_input = _b_sample
    x_sample = Gdec(Genc(xa_sample, is_training=False), att_sample_input, is_training=False)

    # test =========================================================================================================== #
    ckpt_dir = global_args.output_path+'/%s/checkpoints' % global_args.experiment_name

    try:
        tl.load_checkpoint(ckpt_dir, sess)
    except:
        traceback.print_exc()
        sess.close()
        exit()

    try:
        # data for sampling
        contSize = 5
        original_images_list = list()
        original_labels_list = list()
        b_sample_ipt_list = list()

        range_count = 1

        for idx in range(range_count):
            original_images, original_labels = val_data.get_next()
            original_labels_ori = original_labels

            original_images_list.append(original_images)
            original_labels_list.append(original_labels_ori)

            # for absolute
            contList = [-0.5, 0.0, 0.5, 1.0, 1.5]

            intensity_raw = sess.run(data_channel_test.chListTensor, feed_dict={xa_sample: original_images, datacfg.alex_keep_prob: 1.0})
            intensity_raw_list = [intensity_raw + 0.0]
            for i in range(contSize):
                tmps = np.array(intensity_raw, copy=True)
                tmps = tmps + contList[i] * tmps
                intensity_raw_list.append(tmps)

            a_sample_ipt = intensity_raw / max_intensity
            a_sample_ipt = np.expand_dims(a_sample_ipt[:, datacfg.classifierSelectedChannelList[0]], axis=1)

            b_sample_ipt = []

            for i in range(contSize):
                tmp = np.array(a_sample_ipt, copy=True)
                tmp = tmp + contList[i] * tmp
                b_sample_ipt.append(tmp)
            b_sample_ipt_list.append(b_sample_ipt)

        it_per_epoch = 1

        for it in tqdm(range(1)):
            with pylib.Timer(is_output=False) as t:
                epoch = it // it_per_epoch
                it_in_epoch = it % it_per_epoch + 1

                # draw test image
                draw_test_image_digits(global_args, datacfg, epoch, it, it_in_epoch, it_per_epoch,
                                       original_labels_list, original_images_list, b_sample_ipt_list,
                                       sess, colorChannel, x_sample, xa_sample, _b_sample, data_channel_test)
    except:
        traceback.print_exc()
    finally:
        sess.close()
