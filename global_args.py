import pylib
import json


class global_args:
    def __make_dir(self, output_path, name_dir, args):
        pylib.mkdir(output_path + '/%s' % name_dir)
        with open(output_path + '/%s/setting.txt' % name_dir, 'w') as f:
            f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    def __init__(self, args):
        self._atts = args.atts
        self._n_att = len(self._atts)
        self._img_size = args.img_size
        self._shortcut_layers = args.shortcut_layers
        self._inject_layers = args.inject_layers
        self._enc_dim = args.enc_dim
        self._dec_dim = args.dec_dim
        self._dis_dim = args.dis_dim
        self._dis_fc_dim = args.dis_fc_dim
        self._enc_layers = args.enc_layers
        self._dec_layers = args.dec_layers
        self._dis_layers = args.dis_layers

        # training
        self._mode = args.mode
        self._epoch = args.epoch
        self._batch_size = args.batch_size
        self._lr_base = args.lr
        self._n_d = args.n_d
        self._b_distribution = args.b_distribution
        self._thres_int = args.thres_int
        self._test_int = args.test_int
        self._n_sample = args.n_sample

        # others
        self._use_cropped_img = args.use_cropped_img
        self._experiment_name = args.experiment_name
        self._data_name = args.data_name
        self._hyper_param = args.hyper_param
        self._hyper_param2 = args.hyper_param2
        self._description = args.description
        self._classifierSelectedChannel = args.classifierSelectedChannel - 1
        self._classifierSelectedLayer = args.classifierSelectedLayer - 1
        self._trainClassification = args.trainClassification
        self._experiment_method = args.experiment_method

        self._output_path = args.output_path

        self.__make_dir(self._output_path, self._experiment_name, args)

    @property
    def atts(self):
        return self._atts

    @atts.setter
    def atts(self, atts):
        self._atts = atts

    @property
    def n_att(self):
        return self._n_att

    @n_att.setter
    def n_att(self, n_att):
        self._n_att = n_att

    @property
    def img_size(self):
        return self._img_size

    @property
    def shortcut_layers(self):
        return self._shortcut_layers

    @property
    def inject_layers(self):
        return self._inject_layers

    @property
    def enc_dim(self):
        return self._enc_dim

    @enc_dim.setter
    def enc_dim(self, penc_dim):
        self._enc_dim = penc_dim

    @property
    def dec_dim(self):
        return self._dec_dim

    @dec_dim.setter
    def dec_dim(self, pdec_dim):
        self._dec_dim = pdec_dim

    @property
    def dis_dim(self):
        return self._dis_dim

    @dis_dim.setter
    def dis_dim(self, pdis_dim):
        self._dis_dim = pdis_dim

    @property
    def dis_fc_dim(self):
        return self._dis_fc_dim

    @property
    def enc_layers(self):
        return self._enc_layers

    @enc_layers.setter
    def enc_layers(self, penc_layers):
        self._enc_layers = penc_layers

    @property
    def dec_layers(self):
        return self._dec_layers

    @dec_layers.setter
    def dec_layers(self, pdec_layers):
        self._dec_layers = pdec_layers

    @property
    def dis_layers(self):
        return self._dis_layers

    @dis_layers.setter
    def dis_layers(self, pdis_layers):
        self._dis_layers = pdis_layers

    @property
    def mode(self):
        return  self._mode

    @property
    def epoch(self):
        return self._epoch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def lr_base(self):
        return self._lr_base

    @property
    def n_d(self):
        return self._n_d

    @property
    def b_distribution(self):
        return self._b_distribution

    @property
    def thres_int(self):
        return self._thres_int

    @property
    def test_int(self):
        return self._test_int

    @property
    def n_sample(self):
        return self._n_sample

    @property
    def use_cropped_img(self):
        return self._use_cropped_img

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def data_name(self):
        return self._data_name

    @property
    def hyper_param(self):
        return self._hyper_param

    @property
    def hyper_param2(self):
        return self._hyper_param2

    @property
    def description(self):
        return self._description

    @property
    def classifierSelectedChannel(self):
        return self._classifierSelectedChannel

    @property
    def classifierSelectedLayer(self):
        return self._classifierSelectedLayer

    @property
    def trainClassification(self):
        return self._trainClassification

    @property
    def experiment_method(self):
        return self._experiment_method

    @property
    def output_path(self):
        return self._output_path
