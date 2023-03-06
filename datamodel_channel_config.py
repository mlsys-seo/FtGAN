from alexnet import *
import tensorflow as tf


class dataModelChannel:
    def __init__(self):
        self._randomNoiseRange = 1.0

        self._alexInputTensor = None
        self._vggInputTensor = None
        self._vggTriggerInputTensor = None
        self._layerListTensor = None
        self._chListTensor = None
        self._chSelectedTensor = None
        self._randomNoiseAddInputTensor = None
        self._randomNoiseAddInputTensorList = list()
        self._alexnetLogit = None
        self._cost = None
        self._gan_lambda = 1
        self._reconstruction_loss = 10
        self._randomNoise = None
        self._realfake = None
        self._r_cont_mu = None
        self._r_cont_var = None
        self._chatt = None
        self._selected_ch_gradient = None

    # (batch, channeloutput count)
    def extractChannelOutput(self, alex_keep_prob, alexNumLabel, classifierSelectedLayer, data_name):
        logit, self._layerListTensor, _, _ = alexnet(self._alexInputTensor, alex_keep_prob, data_name)
        self._chListTensor = tf.reduce_sum(self._layerListTensor[classifierSelectedLayer], axis=[1, 2])
        self._alexnetLogit = logit

    def extractSelectedChannel(self, channelIndexList):
        channelIndex = channelIndexList[0]
        self._chSelectedTensor = tf.expand_dims(self._chListTensor[:, channelIndex], axis=1)

    def trainClassifier(self, y):
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._alexnetLogit, labels=y))

    def makeRandomNoiseAddInput(self, channelIndexList, min_range=-0.67, max_range=2):
        for channelIndex in channelIndexList:
            frontInputPadding = self.chListTensor[:, : channelIndex]
            backInputPadding = self.chListTensor[:, channelIndex + 1: ]

            minval = self._chListTensor[:, channelIndex] * min_range
            maxval = self._chListTensor[:, channelIndex] * max_range

            randomNoise = tf.random_uniform(tf.shape(self._chListTensor[:, channelIndex]), minval=minval, maxval=maxval)

            self._randomNoiseAddInputTensor = self._chListTensor[:, channelIndex] + randomNoise
            self._randomNoiseAddInputTensor = tf.expand_dims(self._randomNoiseAddInputTensor, axis=1)
            self._randomNoiseAddInputTensor = tf.concat([frontInputPadding,
                                                         self._randomNoiseAddInputTensor,
                                                         backInputPadding], axis=1)

            self._randomNoiseAddInputTensorList.append(self._randomNoiseAddInputTensor)

    def resizeInputTensor(self, resize = False, xa_width = None, xa_height = None):
        if resize:
            if self.alexInputTensor is not None:
                self._alexInputTensor = tf.image.resize_images(self._alexInputTensor, [xa_width, xa_height])

    def setAlexInputTensor(self, xa):
        self._alexInputTensor = xa

    # getter and setter
    @property
    def randomNoiseRange(self):
        return self._randomNoiseRange

    @randomNoiseRange.setter
    def randomNoiseRange(self, range):
        self._randomNoiseRange = range

    @property
    def alexInputTensor(self):
        return self._alexInputTensor

    @alexInputTensor.setter
    def alexInputTensor(self, xa):
        self._alexInputTensor = (xa + 1.0) / 2.0

    @property
    def layerListTensor(self):
        return self._layerListTensor

    @property
    def chListTensor(self):
        return self._chListTensor

    @property
    def randomNoiseAddInputTensorList(self):
        return self._randomNoiseAddInputTensorList

    @property
    def alexnetLogit(self):
        return self._alexnetLogit

    @property
    def randomNoise(self):
        return self._randomNoise

    @property
    def realfake(self):
        return self._realfake

    @property
    def r_cont_mu(self):
        return self._r_cont_mu

    @property
    def r_cont_var(self):
        return self._r_cont_var

    @property
    def cost(self):
        return self._cost

    @property
    def chSelectedTensor(self):
        return self._chSelectedTensor

    @property
    def chatt(self):
        return self._chatt

