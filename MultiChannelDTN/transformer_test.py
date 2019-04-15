# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from MultiChannelDTN.hyperparams import Hyperparams as hp
from MultiChannelDTN.data_load import load_test_data, load_vocab
from MultiChannelDTN.transformer_train import Transformer_Graph

def test():
    # Load graph
    g = Transformer_Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Y = load_test_data()

    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)

    word2idx, idx2word = load_vocab()
     
#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            avg_acc = .0
            for i in range(len(X) // hp.batch_size):

                ### Get mini-batches
                x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                y = Y[i*hp.batch_size: (i+1)*hp.batch_size]

                _preds, _true_preds, _acc, _loss = sess.run([g.preds, g.true_preds, g.acc, g.loss], {g.x: x, g.y: y})
                res_preds = zip(_preds, _true_preds)
                print([(_pred, _true_pred) for _pred, _true_pred in res_preds if (_pred != 0 and _true_pred)])
                print(_acc, _loss)
                avg_acc += _acc
            avg_acc /= (len(X) // hp.batch_size)
            print(avg_acc)
                     

                                          
if __name__ == '__main__':
    test()
    print("Done")
    
    