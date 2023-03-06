from tqdm import tqdm
import numpy as np
import os
import csv

# average
def avg_chIntensity(global_args, sess, datacfg, val_data_classifier, selectedlayer, selectedChannel):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample / 5))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.average(layer1_tot, axis=0)
    intensityLossLambda = 1 / layer1_final[selectedChannel]

    return intensityLossLambda


def avg_chIntensity_carla(global_args, sess, datacfg, val_data_classifier, selectedChannel, xa, data_channel):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        layerList = sess.run([data_channel.layerListTensor], feed_dict={xa: batch_x,
                                                                        data_channel._input_data[1]: batch_y_speed_test,
                                                                        data_channel._dout: [1] * len(data_channel.dropout_vec)})

        layer1 = layerList[0][global_args.classifierSelectedLayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.average(layer1_tot, axis=0)
    intensityLossLambda = 1 / layer1_final[selectedChannel]

    return intensityLossLambda

# max intensity find
def max_chIntensity_digits(global_args, sess, datacfg, val_data_classifier, selectedlayer, selectedChannel):
    import tensorflow as tf
    max_intensity = 0

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample / 5))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        batch_x = tf.image.resize_images(batch_x, (32, 32))
        batch_x = sess.run(batch_x)

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)[:, selectedChannel]
        layer1_max = np.max(layer1_sum)
        if layer1_max > max_intensity:
            max_intensity = layer1_max

    max_intensity = max_intensity + (max_intensity * 0.2)

    return max_intensity


def max_chIntensity(global_args, sess, datacfg, tr_data, selectedlayer, selectedChannel):
    max_intensity = 0

    for batchIndex in tqdm(range(int(len(tr_data) / global_args.n_sample / 5))):
        batch = tr_data.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)[:, selectedChannel]

        layer1_max = np.max(layer1_sum)
        if layer1_max > max_intensity:
            max_intensity = layer1_max

    max_intensity = max_intensity + (max_intensity * 0.2)

    return max_intensity

def max_chIntensity_vgg(global_args, sess, datacfg, tr_data, selectedlayer, selectedChannel):
    max_intensity = 0

    for batchIndex in tqdm(range(int(len(tr_data) / global_args.n_sample / 5))):
        batch = tr_data.get_next()

        batch_x = batch[0]
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)[:, selectedChannel]

        layer1_max = np.max(layer1_sum)
        if layer1_max > max_intensity:
            max_intensity = layer1_max

    max_intensity = max_intensity + (max_intensity * 0.2)

    return max_intensity


def max_chIntensity_vggface(global_args, sess, datacfg, tr_data, selectedlayer, selectedChannel):
    max_intensity = 0

    def preprocess(image):
        averageImage = [129.1863, 104.7624, 93.5940]
        # data = np.float32(np.moveaxis(pix, 2, 0))
        data = np.float32(np.moveaxis(image, 3, 0)[::-1])
        data[0] -= averageImage[2]
        data[1] -= averageImage[1]
        data[2] -= averageImage[0]
        data = np.moveaxis(data, 0, 3)
        return data

    for batchIndex in tqdm(range(int(len(tr_data) / global_args.n_sample / 5))):
        batch = tr_data.get_next()

        batch_x = (batch[0] - batch[0].min()) * 255 / (batch[0].max() - batch[0].min())
        batch_x = preprocess(batch_x)
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)[:, selectedChannel]

        layer1_max = np.max(layer1_sum)
        if layer1_max > max_intensity:
            max_intensity = layer1_max

    max_intensity = max_intensity + (max_intensity * 0.2)

    return max_intensity


def max_chIntensity_carla(global_args, sess, datacfg, tr_data, selectedChannel, xa, data_channel):
    max_intensity = 0

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(tr_data[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        layerList = sess.run([data_channel.layerListTensor], feed_dict={xa: batch_x,
                                                                        data_channel._input_data[1]: batch_y_speed_test,
                                                                        data_channel._dout: [1] * len(data_channel.dropout_vec)})

        layer1 = layerList[0][global_args.classifierSelectedLayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)[:, selectedChannel]

        layer1_max = np.max(layer1_sum)
        if layer1_max > max_intensity:
            max_intensity = layer1_max

    max_intensity = max_intensity + (max_intensity * 0.2)

    return max_intensity



def check_chIntensity_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.average(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_chGradient_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    import tensorflow as tf

    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        grad = tf.gradients(datacfg.alex_logit, datacfg.alex_layerList[selectedlayer])
        grad_result = sess.run(grad, feed_dict={datacfg.alex_x: batch_x, datacfg.alex_y: batch_y, datacfg.alex_keep_prob: 1.0})[0]

        layer1_sum = np.abs(np.sum(np.sum(grad_result, axis=1), axis=1))
        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.average(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_activation_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    import tensorflow as tf

    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})
        layer1 = layerList[0][selectedlayer]
        activation_count = np.count_nonzero(layer1, axis=(1, 2))

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(activation_count)
        else:
            layer1_tot = np.append(layer1_tot, activation_count, axis=0)

    layer1_final = np.average(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)

# Max & count

def check_chIntensityMax_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.max(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_chGradientMax_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    import tensorflow as tf

    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        grad = tf.gradients(datacfg.alex_logit, datacfg.alex_layerList[selectedlayer])
        grad_result = sess.run(grad, feed_dict={datacfg.alex_x: batch_x, datacfg.alex_y: batch_y, datacfg.alex_keep_prob: 1.0})[0]

        layer1_sum = np.abs(np.sum(np.sum(grad_result, axis=1), axis=1))
        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.max(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_chActivationCount_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]

        activation_count_sum = np.max(np.count_nonzero(layer1, axis=(1, 2)), axis=0)
        activation_count_sum = np.expand_dims(activation_count_sum, axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.expand_dims(np.append(layer1_tot, activation_count_sum), axis=1)
        else:
            layer1_tot = np.append(layer1_tot, activation_count_sum, axis=1)

    layer1_final_activated = np.max(layer1_tot, axis=1)
    layer1_final_top = np.argsort(layer1_final_activated)[-5:]
    layer1_final_bottom = np.argsort(layer1_final_activated)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


# variance

def check_chIntensity_variance_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.var(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_gradient_variance_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    import tensorflow as tf

    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        grad = tf.gradients(datacfg.alex_logit, datacfg.alex_layerList[selectedlayer])
        grad_result = sess.run(grad, feed_dict={datacfg.alex_x: batch_x, datacfg.alex_y: batch_y, datacfg.alex_keep_prob: 1.0})[0]

        layer1_sum = np.abs(np.sum(np.sum(grad_result, axis=1), axis=1))
        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.var(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_chActivation_variance_rank(global_args, sess, datacfg, val_data_classifier, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(int(len(val_data_classifier) / global_args.n_sample))):
        batch = val_data_classifier.get_next()

        batch_x = (batch[0] + 1.0) / 2.0
        batch_y = batch[1]

        layerList = sess.run([datacfg.alex_layerList], feed_dict={datacfg.alex_x: batch_x,
                                                                  datacfg.alex_y: batch_y,
                                                                  datacfg.alex_keep_prob: 1.0})

        layer1 = layerList[0][selectedlayer]

        activation_count_sum = np.max(np.count_nonzero(layer1, axis=(1, 2)), axis=0)
        activation_count_sum = np.expand_dims(activation_count_sum, axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.expand_dims(np.append(layer1_tot, activation_count_sum), axis=1)
        else:
            layer1_tot = np.append(layer1_tot, activation_count_sum, axis=1)

    layer1_final = np.var(layer1_tot, axis=1)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)

# carla

def check_chIntensity_rank_carla(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        layerList = sess.run([data_channel.layerListTensor],
                             feed_dict={xa: batch_x,
                                        data_channel._input_data[1]: batch_y_speed_test,
                                        data_channel._dout: [1] * len(data_channel.dropout_vec)})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.average(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_gradient_rank_carla(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    import tensorflow as tf

    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        grad = tf.gradients(data_channel.alexnetLogit, data_channel.layerListTensor[selectedlayer])
        grad_result = sess.run(grad, feed_dict={xa: batch_x,
                                                data_channel._input_data[1]: batch_y_speed_test,
                                                data_channel._dout: [1] * len(data_channel.dropout_vec)})[0]

        layer1_sum = np.abs(np.sum(np.sum(grad_result, axis=1), axis=1))
        # layer1_avg = np.sum(layer1_sum, axis=0) / 10
        # layer1_avg = np.expand_dims(layer1_avg, axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.average(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_activation_rank_carla(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        layerList = sess.run([data_channel.layerListTensor],
                             feed_dict={xa: batch_x,
                                        data_channel._input_data[1]: batch_y_speed_test,
                                        data_channel._dout: [1] * len(data_channel.dropout_vec)})

        layer1 = layerList[0][selectedlayer]
        activation_count = np.count_nonzero(layer1, axis=(1, 2))

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(activation_count)
        else:
            layer1_tot = np.append(layer1_tot, activation_count, axis=0)

    layer1_final = np.average(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)

# carla Max & Count

def check_chIntensityMax_rank_carla(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        layerList = sess.run([data_channel.layerListTensor],
                             feed_dict={xa: batch_x,
                                        data_channel._input_data[1]: batch_y_speed_test,
                                        data_channel._dout: [1] * len(data_channel.dropout_vec)})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.max(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_chGradientMax_rank_carla(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    import tensorflow as tf

    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        grad = tf.gradients(data_channel.alexnetLogit, data_channel.layerListTensor[selectedlayer])
        grad_result = sess.run(grad, feed_dict={xa: batch_x,
                                                data_channel._input_data[1]: batch_y_speed_test,
                                                data_channel._dout: [1] * len(data_channel.dropout_vec)})[0]
        layer1_sum = np.abs(np.sum(np.sum(grad_result, axis=1), axis=1))
        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.max(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_chActivationCount_rank_carla(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        layerList = sess.run([data_channel.layerListTensor],
                             feed_dict={xa: batch_x,
                                        data_channel._input_data[1]: batch_y_speed_test,
                                        data_channel._dout: [1] * len(data_channel.dropout_vec)})

        layer1 = layerList[0][selectedlayer]

        activation_count_sum = np.max(np.count_nonzero(layer1, axis=(1, 2)), axis=0)
        activation_count_sum = np.expand_dims(activation_count_sum, axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.expand_dims(np.append(layer1_tot, activation_count_sum), axis=1)
        else:
            layer1_tot = np.append(layer1_tot, activation_count_sum, axis=1)

    layer1_final = np.max(layer1_tot, axis=1)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)

# carla variance

def check_chIntensity_variance_rank_carla(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        layerList = sess.run([data_channel.layerListTensor],
                             feed_dict={xa: batch_x,
                                        data_channel._input_data[1]: batch_y_speed_test,
                                        data_channel._dout: [1] * len(data_channel.dropout_vec)})

        layer1 = layerList[0][selectedlayer]
        layer1_sum = np.sum(np.sum(layer1, axis=1), axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.var(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_gradient_variance_rank_carla(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    import tensorflow as tf

    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        grad = tf.gradients(data_channel.alexnetLogit, data_channel.layerListTensor[selectedlayer])
        grad_result = sess.run(grad, feed_dict={xa: batch_x,
                                                data_channel._input_data[1]: batch_y_speed_test,
                                                data_channel._dout: [1] * len(data_channel.dropout_vec)})[0]

        layer1_sum = np.abs(np.sum(np.sum(grad_result, axis=1), axis=1))
        # layer1_avg = np.sum(layer1_sum, axis=0) / 10
        # layer1_avg = np.expand_dims(layer1_avg, axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.copy(layer1_sum)
        else:
            layer1_tot = np.append(layer1_tot, layer1_sum, axis=0)

    layer1_final = np.var(layer1_tot, axis=0)
    layer1_final_top = np.argsort(layer1_final)[-5:]
    layer1_final_bottom = np.argsort(layer1_final)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)


def check_chActivationCount_rank_carla2(global_args, sess, datacfg, val_data_classifier, data_channel, xa, selectedlayer):
    layer1_tot = np.array([])

    for batchIndex in tqdm(range(374)):
        batch_x, batch_y = next(val_data_classifier[datacfg.carla_branchIndex])
        batch_y_speed = batch_y[:, 10]
        batch_y_speed_test = np.array(batch_y_speed / 25.0)
        batch_y_speed_test = batch_y_speed_test.reshape((len(batch_y), 1))

        layerList = sess.run([data_channel.layerListTensor],
                             feed_dict={xa: batch_x,
                                        data_channel._input_data[1]: batch_y_speed_test,
                                        data_channel._dout: [1] * len(data_channel.dropout_vec)})

        layer1 = layerList[0][selectedlayer]

        activation_count_sum = np.max(np.count_nonzero(layer1, axis=(1, 2)), axis=0)
        activation_count_sum = np.expand_dims(activation_count_sum, axis=1)

        if len(layer1_tot) == 0:
            layer1_tot = np.expand_dims(np.append(layer1_tot, activation_count_sum), axis=1)
        else:
            layer1_tot = np.append(layer1_tot, activation_count_sum, axis=1)

    layer1_final_activated = np.max(layer1_tot, axis=1)
    layer1_final_top = np.argsort(layer1_final_activated)[-5:]
    layer1_final_bottom = np.argsort(layer1_final_activated)[: 5]

    print("Top    5 : ", np.flip(layer1_final_top))
    print("Bottom 5 : ", layer1_final_bottom)
