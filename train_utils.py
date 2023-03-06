import numpy as np
import csv
import os
import cv2

import pylib
import imlib as im

from tqdm import tqdm
from sklearn.utils.extmath import softmax
import data

def check_to_output(global_args, epoch, it, it_per_epoch):
    result = True

    if (global_args._experiment_method == "train"):
        if not (((epoch + 1) % 2 == 0) and (it % it_per_epoch == 0)):
            result = False
        test_dir = ''
    else:
        test_dir = '/sample_test_data/'

    return result, test_dir


def display(epoch, it, it_in_epoch, it_per_epoch, t):
    if it % it_per_epoch == 0:
        print("Epoch: (%3d) (%5d/%5d) Time: %s!" % (epoch, it_in_epoch, it_per_epoch, t))


def save(epoch, it, it_in_epoch, it_per_epoch, saver, sess, ckpt_dir):
    if (epoch % 2 == 0) and (it % it_per_epoch == 0) and (epoch != 0):
        save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
        print('Model is saved at %s!' % save_path)


def draw_test_image_digits(global_args, datacfg, epoch, it, it_in_epoch, it_per_epoch,
                           original_labels_ori, original_images_list, b_sample_ipt_list,
                           sess, colorChannel, x_sample, xa_sample, _b_sample, data_channel_test):

    result, test_dir = check_to_output(global_args, epoch, it, it_per_epoch)
    if not result:
        return

    for image_range in range(len(original_labels_ori)):
        indexList = list()
        original_images = original_images_list[image_range]

        # for whole data
        for _ in range(100):
            indexList.append(_)

        # Add black line
        x_sample_opt_list = [original_images[indexList], np.full((len(indexList),
                                                                global_args.img_size,
                                                                global_args.img_size // 10, colorChannel), -1.0)]

        b_sample_ipt_list_extract = list()
        for idx in range(len(b_sample_ipt_list[image_range])):
            b_sample_ipt_list_extract.append(b_sample_ipt_list[image_range][idx][indexList]) #for absolute

        for i, b_sample_ipt in enumerate(b_sample_ipt_list_extract):
            ganImList = sess.run(x_sample, feed_dict={xa_sample: original_images[indexList], _b_sample: b_sample_ipt})
            x_sample_opt_list.append(ganImList)

        save_dir = global_args.output_path + '/%s/' % global_args.experiment_name + test_dir
        pylib.mkdir(save_dir)

        sample = np.concatenate(x_sample_opt_list, 2)
        im.imwrite(im.immerge(sample, len(indexList), 1), '%s/test[0-99].png' %(save_dir))
