from alexnet import *
import random

class dataModelConfig:
    def __init__(self, inputSize, classifierSelectedChannel, classifierSelectedLayer=1, alexNumLabel=10, n_att=1):
        self._inputSize = inputSize
        self._classifierSelectedChannelList = list()
        self._classifierSelectedLayer = classifierSelectedLayer
        self._alexNumLabel = alexNumLabel

        self._alex_x = None
        self._alex_y = None
        self._alex_keep_prob = None
        self._alex_logit = None
        self._alex_layerList = None
        self._alex_weights = None
        self._alex_accuracy = None
        self._layerListNoRelu = None

        self._n_att = n_att
        self._resize = False

        self._channelLen = 1


    def SetClassifierSelectedChannelList(self, classifierSelectedChannel):
        self._classifierSelectedChannelList.append(classifierSelectedChannel)

    #Alexnet use functions
    def LoadAlexStructure(self, dataset, description="_"):
        self._alex_x, \
        self._alex_y, \
        self._alex_keep_prob, \
        self._alex_logit, \
        self._alex_layerList, \
        self._alex_weights, \
        self._alex_accuracy, \
        self._layerListNoRelu = \
            loadAlexStructure(dataset, description)

    def LoadAlexWeight(self, sess, dataset, description="_"):
        loadAlexWeight(sess, dataset, description)

    def ReturnAlexVariable(self):
        return self._alex_x, \
               self._alex_y, \
               self._alex_keep_prob, \
               self._alex_logit, \
               self._alex_layerList, \
               self._alex_weights, \
               self._alex_accuracy, \
               self._layerListNoRelu

    #getter and setter
    @property
    def inputSize(self):
        return self._inputSize

    @inputSize.setter
    def inputSize(self, inputSize):
        self._inputSize = inputSize

    @property
    def classifierSelectedChannelList(self):
        return self._classifierSelectedChannelList

    @classifierSelectedChannelList.setter
    def classifierSelectedChannelList(self, classifierSelectedChannelList):
        self._classifierSelectedChannelList = classifierSelectedChannelList

    @property
    def classifierSelectedLayer(self):
        return self._classifierSelectedLayer

    @classifierSelectedLayer.setter
    def classifierSelectedLayer(self, classifierSelectedLayer):
        self._classifierSelectedLayer = classifierSelectedLayer

    @property
    def alexNumLabel(self):
        return self._alexNumLabel

    @alexNumLabel.setter
    def alexNumLabel(self, alexNumLabel):
        self._alexNumLabel = alexNumLabel

    @property
    def n_att(self):
        return self._n_att

    @n_att.setter
    def n_att(self, n_att):
        self._n_att = n_att

    @property
    def resize(self):
        return self._resize

    @resize.setter
    def resize(self, gan_img_size):
        if self._inputSize != gan_img_size:
            self._resize = True

    @property
    def channelLen(self):
        return self._channelLen

    @property
    def Xaxis(self):
        return self._xaxis

    @property
    def Yaxis(self):
        return self._yaxis

    @property
    def alex_x(self):
        return self._alex_x

    @property
    def alex_y(self):
        return self._alex_y

    @property
    def alex_keep_prob(self):
        return self._alex_keep_prob

    @property
    def alex_logit(self):
        return self._alex_logit

    @property
    def alex_layerList(self):
        return self._alex_layerList

    @property
    def alex_weights(self):
        return self._alex_weights

    @property
    def alex_accuracy(self):
        return self._alex_accuracy

    @property
    def layerListNoRelu(self):
        return self._layerListNoRelu
