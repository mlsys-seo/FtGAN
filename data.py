from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from tflib.utils import session

from alexnet import *

import keras
from keras.datasets import mnist


def batch_dataset(dataset, batch_size, prefetch_batch=2, drop_remainder=True, filter=None,
                  map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset


def disk_image_batch_dataset(img_paths, batch_size, labels=None, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, useNumpy=False, sess=None):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    elif isinstance(labels, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
    else:

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def parse_func(path, *label):
        if useNumpy:
            return (path,) + label


        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 3)

        return (img,) + label


    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    # dataset = dataset.map(parse_func, num_parallel_calls=num_threads) is slower

    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat)

    return dataset


class Dataset(object):

    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

        self._is_eager = tf.executing_eagerly()
        #self._is_eager = False
        self._eager_iterator = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        if self._is_eager:
            return self._eager_iterator.get_next()
        else:
            return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        if self._is_eager:
            self._eager_iterator = tf.data.Iterator(self._dataset)#tfe.Iterator(self._dataset)
        else:
            self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):
        self._dataset = dataset

        if self._is_eager:
            self._eager_iterator = tf.data.Iterator(self._dataset)#tfe.Iterator(dataset)
        else:
            self._iterator = dataset.make_initializable_iterator()
            self._batch_op = self._iterator.get_next()
            if sess:
                self._sess = sess
            else:
                self._sess = session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op


class Mnist(Dataset):
    def __init__(self, data_dir, atts, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, sess=None, part='train', crop=True,
                 experiment_name="",  global_args=None, select_label=None):
        super(Mnist, self).__init__()

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        def _map_func(img, label):
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1

            return img, label

        if part == 'test' or part == 'val':
            drop_remainder = False
            shuffle = False
            repeat = 1
            img_paths = x_test
            labelsCat = y_test
        else:
            img_paths = x_train
            labelsCat = y_train

        self.real_image = img_paths
        self._img_num = len(img_paths)

        num_classes = 10
        labelsCat = keras.utils.to_categorical(labelsCat, num_classes)
        labels = labelsCat

        img_paths = np.reshape(np.array(img_paths), [img_paths.shape[0], img_paths.shape[1], img_paths.shape[2], 1])

        dataset = disk_image_batch_dataset(img_paths=img_paths,
                                           labels=labels,
                                           batch_size=batch_size,
                                           prefetch_batch=prefetch_batch,
                                           drop_remainder=drop_remainder,
                                           map_func=_map_func,
                                           num_threads=num_threads,
                                           shuffle=shuffle,
                                           buffer_size=buffer_size,
                                           repeat=repeat,
                                           useNumpy=True)

        self._bulid(dataset, sess)

    def __len__(self):
        return self._img_num
