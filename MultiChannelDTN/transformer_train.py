# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from MultiChannelDTN.hyperparams import Hyperparams as hp
from MultiChannelDTN.data_load import get_batch_data, load_vocab
from MultiChannelDTN.modules import *


class Transformer_Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # if is_training:
            #     self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
            # else:  # inference
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.class_num))

            # Load vocabulary
            word2idx, idx2word = load_vocab()

            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc = embedding(self.x,
                                     vocab_size=len(word2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     zero_pad=False,
                                     scope="enc_embed")

                key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1)

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe")
                else:
                    self.enc += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")


                self.enc *= key_masks

                ## Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False)
                        self.mutihead = self.enc
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])
            # Final linear projection
            self.logits = tf.layers.dense(self.enc, hp.class_num)
            self.logits = tf.reshape(self.logits, [-1, hp.maxlen * hp.class_num])
            self.logits = tf.layers.dense(self.logits, hp.class_num)

            self.preds = tf.to_int32(tf.arg_max(tf.nn.softmax(self.logits, axis=-1), dimension=-1))
            self.y_label = tf.to_int32(tf.argmax(self.y, -1))
            self.true_preds = tf.equal(self.preds, self.y_label)

            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y_label, self.preds), tf.float32))

            self.y_smoothed = label_smoothing(tf.cast(self.y, dtype=tf.float32))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.loss = tf.reduce_mean(self.loss)
            if is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('mean_loss', self.loss)
                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # Load vocabulary
    word2idx, idx2word = load_vocab()

    # Construct graph
    g = Transformer_Graph(is_training=True)
    print("Graph loaded")

    # Start session
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop(): break
            for x, y in get_batch_data():
                try:
                    _preds = sess.run([g.preds], {g.x: x, g.y: y})
                    # print(_preds)
                    _acc, _loss, _ = sess.run([g.acc, g.loss, g.train_op], {g.x: x, g.y: y})
                    print(_acc, _loss)
                except Exception as e:
                    print(x, y)
            try:
                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
            except Exception as e:
                print(epoch, gs)
    print("Done")