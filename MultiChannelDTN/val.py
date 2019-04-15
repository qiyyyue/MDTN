from __future__ import print_function

import os

import tensorflow as tf

from MultiChannelDTN.hyperparams import Hyperparams as hp
from MultiChannelDTN.data_load import load_val_data, load_test_data
from MultiChannelDTN.modules import *
from MultiChannelDTN.MCDTN_Model import multi_channel_dtn

save_dir = '../CheckPionts/MTDN/MTDN_100/model_100'

def val():
    model = multi_channel_dtn(False)
    print("Graph loaded")

    hp.use_kg_embd = True
    # Load data
    trans_X, kg_X, Y = load_test_data()

    indices = np.random.permutation(np.arange(len(trans_X)))
    trans_X = trans_X[indices]
    kg_X = kg_X[indices]
    Y = Y[indices]

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_dir)

    sum_acc = .0
    for i in range(len(trans_X) // hp.batch_size):
        ### Get mini-batches
        trans_x = trans_X[i * hp.batch_size: (i + 1) * hp.batch_size]
        kg_x = kg_X[i * hp.batch_size: (i + 1) * hp.batch_size]
        y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]

        _acc = session.run(model.acc, {model.trans_x: trans_x, model.kg_x: kg_x, model.y: y})
        print(_acc)
        sum_acc += _acc

    print('val acc is {:.4f}'.format(sum_acc/(len(trans_X)//hp.batch_size)))




val()
print("Done")