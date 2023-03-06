import tensorflow as tf
import tensorflow.contrib.layers as layers


def conv2d(x, W, strides=1, name=""):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', name=name + "_mat")
    return tf.nn.relu(x, name=name+"_relu"), x


def maxpool2d(x, k=2,name=""):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME',name=name+"_max")


def minibatch(input, num_kernels=10, kernel_dim=3, name=None):
    output_dim = num_kernels * kernel_dim
    w = tf.get_variable("Weight_minibatch_", [input.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(stddev=0.2))
    b = tf.get_variable("Bias_minibatch_", [output_dim], initializer=tf.constant_initializer(0.0))
    x = tf.matmul(input, w) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)

    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    output = tf.concat([input, minibatch_features], 1)

    return output


# Session setting
# Create model
def alexnet(x, keep_prob, description="_", for_adv=False):
    #inputSize
    numLabel = 10

    with tf.variable_scope('classification', reuse=tf.AUTO_REUSE):
        convFinalOutputWidth = int(int(x.shape[1]) / 4)
        convFinalOutputHeight = int(int(x.shape[2]) / 4)

        initializer = tf.contrib.layers.xavier_initializer()
        # Weight and Bias

        W_conv1 = tf.get_variable("W_conv1", shape=[5, 5, x.shape.as_list()[3], 96], initializer=initializer)
        W_conv2 = tf.get_variable("W_conv2", shape=[5, 5, 96, 256], initializer=initializer)
        W_conv3 = tf.get_variable("W_conv3", shape=[3, 3, 256, 384], initializer=initializer)
        W_conv4 = tf.get_variable("W_conv4", shape=[3, 3, 384, 384], initializer=initializer)
        W_conv5 = tf.get_variable("W_conv5", shape=[3, 3, 384, 256], initializer=initializer)

        W_fc1 = tf.get_variable("W_fc1", shape=[convFinalOutputWidth * convFinalOutputHeight * 256, 512], initializer=initializer)
        W_fc2 = tf.get_variable("W_fc2", shape=[512, 512], initializer=initializer)
        W_fc3 = tf.get_variable("W_fc3", shape=[512, numLabel], initializer=initializer)

        weights = { 'conv1': W_conv1,
                    'conv2': W_conv2,
                    'conv3': W_conv3,
                    'conv4': W_conv4,
                    'conv5': W_conv5,
                    'fc1':   W_fc1,
                    'fc2':   W_fc2,
                    'fc3':   W_fc3,}

        # Reshape input picture
        # Convolution Layer
        conv1, conv1NoRelu = conv2d(x, weights['conv1'], name="conv1")
        conv1max = maxpool2d(conv1, k=2, name="conv1")

        # Convolution Layer
        conv2, conv2NoRelu = conv2d(conv1max, weights['conv2'], name="conv2")
        conv2max = maxpool2d(conv2, k=2, name="conv2")

        conv3, conv3NoRelu = conv2d(conv2max, weights['conv3'], name="conv3")
        conv4, conv4NoRelu = conv2d(conv3, weights['conv4'], name="conv4")
        conv5, conv5NoRelu = conv2d(conv4, weights['conv5'], name="conv5")

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv5, [-1, convFinalOutputWidth * convFinalOutputHeight * 256], name="fc1_reshape")
        fc1 = tf.matmul(fc1, weights['fc1'], name="fc1_mat")
        fc1 = tf.nn.relu(fc1, name="fc1_relu")
        fc1 = tf.nn.dropout(fc1, keep_prob, name="fc1_drop")

        fc2 = tf.matmul(fc1, weights['fc2'], name="fc2_mat")
        fc2 = tf.nn.relu(fc2, name="fc2_relu")
        fc2 = tf.nn.dropout(fc2, keep_prob, name="fc2_drop")

        # Output, class logitiction
        fc3 = tf.matmul(fc2, weights['fc3'], name="fc3_mat")
        out = fc3

        layerList = [conv1, conv2, conv3, conv4, conv5, fc1, fc2]
        layerListNoRelu = [conv1NoRelu, conv2NoRelu, conv3NoRelu, conv4NoRelu, conv5NoRelu]

    if for_adv:
        return out
    else:
        return out, layerList, weights, layerListNoRelu


def loadAlexStructure(data_name, description="_"):
    alexNumLabel = 10
    inputSize = 32

    x = tf.placeholder(tf.float32, (None, inputSize, inputSize, 1))
    y = tf.placeholder(tf.float32, (None,alexNumLabel))
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Construct model
    logit, layerList, weights, layerListNoRelu = alexnet(x, keep_prob, "mnist")
    output = tf.nn.softmax(logit, name='output')
    correct_logit = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))

    # Evaluate model
    accuracy = tf.reduce_mean(tf.cast(correct_logit, tf.float32))

    return x, y, keep_prob, logit, layerList, weights, accuracy, layerListNoRelu


def loadAlexWeight(sess, data_name, description="_"):
    MODEL_PATH = "./alexnet/"
    MODEL_NAME = "./alexNet_mnist"

    print(MODEL_NAME)

    saver = tf.train.Saver(var_list=tf.trainable_variables(scope='classification'))
    saver.restore(sess, MODEL_PATH + MODEL_NAME+'.ckpt')
