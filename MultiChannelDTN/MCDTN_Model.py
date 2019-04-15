import tensorflow as tf
import numpy as np
import pandas as pd

from MultiChannelDTN.hyperparams import Hyperparams as hp
from MultiChannelDTN.modules import *
from MultiChannelDTN.data_load import load_vocab

class multi_channel_dtn(object):

    def __init__(self, _is_training = True):

        self.trans_x = tf.placeholder(tf.int32, [None, hp.maxlen], 'trans_x')
        self.kg_x = tf.placeholder(tf.float32, [None, hp.maxlen, hp.kg_embd_dim], 'kg_x')
        self.y = tf.placeholder(tf.int32, [None, hp.class_num], 'y')

        self.is_training = _is_training

        self.build_model()

    def build_model(self):

        self.transformer_enc = self.transformer(self.trans_x)
        self.bilstm_enc = self.bi_lstm(self.kg_x)
        self.enc = tf.concat([self.transformer_enc, self.bilstm_enc], axis=-1)

        self.conv1d = self.h_cnn(self.enc)

        self.fc = self.full_connect(self.conv1d)
        # Final linear projection
        self.logits = tf.layers.dense(self.fc, hp.class_num, name='fc2')

        self.preds = tf.to_int32(tf.arg_max(tf.nn.softmax(self.logits, axis=-1), dimension=-1))
        self.y_label = tf.to_int32(tf.argmax(self.y, -1))
        self.true_preds = tf.equal(self.preds, self.y_label)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y_label, self.preds), tf.float32))

        if self.is_training:
            self.y_smoothed = label_smoothing(tf.cast(self.y, dtype=tf.float32))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.loss = tf.reduce_mean(self.loss)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.loss)
            self.merged = tf.summary.merge_all()


    def h_cnn(self, input):
        # conv and max pooling
        conv = conv1d(input, hp.num_filters, hp.cnn_kernel_size, name='conv1d')
        gmp = global_max_pooling(conv, axis=1, name='gmp')

        # re_conv = tf.reshape(conv, [-1, (hp.maxlen-hp.cnn_kernel_size+1)*hp.num_filters])
        return gmp

    # full conect
    def full_connect(self, inputs):
        fc = tf.layers.dense(inputs, hp.fc_hidden_dim, name='fc1')
        fc = tf.contrib.layers.dropout(fc, hp.dropout_rate)
        fc = tf.nn.relu(fc)

        return fc


    def bi_lstm(self, input_x):

        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hp.hidden_units)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hp.hidden_units)

        # used = tf.sign(tf.reduce_max(tf.abs(input_x), 2))
        # seq_len = tf.reduce_sum(used, 1)
        # seq_len = tf.cast(seq_len, tf.int32)

        # dropout
        if self.is_training:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=(1 - hp.dropout_rate))
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=(1 - hp.dropout_rate))

        # bi lstm
        outputs, f_states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            input_x,
            dtype=tf.float32
        )

        # out put of bilstm
        f_output, b_output = outputs
        f_output, b_output = normalize(f_output), normalize(b_output)
        enc = tf.concat([f_output, b_output], -1)

        return enc

    def transformer(self, input_x):
        word2idx, idx2word = load_vocab()

        with tf.variable_scope("encoder"):
            ## Embedding
            enc = embedding(input_x,
                                 vocab_size=len(word2idx),
                                 num_units=hp.hidden_units,
                                 scale=True,
                                 scope="enc_embed")

            key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(enc), axis=-1)), -1)

            ## Positional Encoding
            if hp.sinusoid:
                enc += positional_encoding(input_x,
                                                num_units=hp.hidden_units,
                                                zero_pad=False,
                                                scale=False,
                                                scope="enc_pe")
            else:
                enc += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(input_x)[1]), 0), [tf.shape(input_x)[0], 1]),
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe")

            enc *= key_masks

            ## Dropout
            enc = tf.layers.dropout(enc,
                                         rate=hp.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc = multihead_attention(queries=enc,
                                                   keys=enc,
                                                   num_units=hp.hidden_units,
                                                   num_heads=hp.num_heads,
                                                   dropout_rate=hp.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=False)
                    ### Feed Forward
                    enc = feedforward(enc, num_units=[4 * hp.hidden_units, hp.hidden_units])
        return enc
