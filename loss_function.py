import tensorflow as tf


def makeCustomDLoss(d_loss_gan, gp, xa_loss_att=None, d_ch_loss_gan=None, gp_ch=None):
    d_loss = d_loss_gan + gp * 10.0

    if (d_ch_loss_gan != None) and (gp_ch != None):
        d_loss = d_loss + d_ch_loss_gan  + gp_ch * 10.0

    elif xa_loss_att != None:
        d_loss = d_loss + xa_loss_att

    return d_loss

def make_xbLossAttClassifier(chListTensor, b, classifierSelectedChannel, remainLossCorrection=1.0):
    frontchInputPadding = chListTensor[:, :classifierSelectedChannel]
    backchInputPadding = chListTensor[:, classifierSelectedChannel + 1:]
    chremainPadding = tf.concat([frontchInputPadding, backchInputPadding], axis=1)

    frontbInputPadding = b[:, :classifierSelectedChannel]
    backbInputPadding = b[:, classifierSelectedChannel + 1:]
    bremainPadding = tf.concat([frontbInputPadding, backbInputPadding], axis=1)


    selectLoss = tf.losses.absolute_difference(b[:, classifierSelectedChannel], chListTensor[:, classifierSelectedChannel])
    remainLoss = tf.losses.absolute_difference(bremainPadding, chremainPadding)

    xb__loss_att_classifier = 1.0 * (selectLoss) + remainLossCorrection * (remainLoss)

    return xb__loss_att_classifier


def make_outputLoss(logit, b_logit):
    outputLoss = tf.losses.absolute_difference(logit, b_logit)

    return outputLoss
