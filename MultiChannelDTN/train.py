from __future__ import print_function

import os

import tensorflow as tf

from MultiChannelDTN.hyperparams import Hyperparams as hp
from MultiChannelDTN.data_load import load_train_data, load_vocab, get_batch_data
from MultiChannelDTN.modules import *
from MultiChannelDTN.MCDTN_Model import multi_channel_dtn
from Model.kg_embedding_model.triple2vec import triple2vec

save_dir = '../CheckPionts/MTDN/MTDN_100/model_100'
tensorboard_dir = '../tensorboard/MTDN'

def train():
    # Construct graph
    g = multi_channel_dtn(True)
    print("Graph loaded")

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", g.loss)
    tf.summary.scalar("accuracy", g.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    # trans_X, kg_X, Y = load_train_data()
    for epoch in range(1, hp.num_epochs + 1):

        # indices = np.random.permutation(np.arange(len(trans_X)))
        # trans_X = trans_X[indices]
        # kg_X = kg_X[indices]
        # Y = Y[indices]

        sum_loss = 0
        batch_cnt = 0
        sum_acc = 0.0
        # for i in range(len(trans_X) // hp.batch_size):
        #     ### Get mini-batches
        #     trans_x = trans_X[i * hp.batch_size: (i + 1) * hp.batch_size]
        #     kg_x = kg_X[i * hp.batch_size: (i + 1) * hp.batch_size]
        #     y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]

        for trans_x, kg_x, y in get_batch_data():
            _acc, _loss, _ = sess.run([g.acc, g.loss, g.train_op], {g.trans_x: trans_x, g.kg_x: kg_x, g.y: y})
            #print(_logits.shape)
            #print(_loss.shape)
            sum_loss += _loss
            sum_acc += _acc
            batch_cnt += 1
            print('\tacc:{:.3f}, loss:{:.3f}'.format(_acc, _loss))
        print('epoch {}, avg_loss:{:.3f}, avg_acc:{:.3f}'.format(epoch, sum_loss / batch_cnt, sum_acc/batch_cnt))
#        break

    saver.save(sess=sess, save_path=save_dir)
print("Done")

train()