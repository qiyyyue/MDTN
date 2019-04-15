# -*- coding: utf-8 -*-
'''
'''
import os


class Hyperparams:
    '''Hyperparameters'''
    # data
    data_base_dir = '../Data/train_news_data/2016_election'
    train_path = os.path.join(data_base_dir, 'train.txt')
    test_path = os.path.join(data_base_dir, 'test.txt')
    val_path = os.path.join(data_base_dir, 'val.txt')
    vocab_path = os.path.join(data_base_dir, 'vocab_cnt.txt')

    use_kg_embd = True
    kg_embd_dim = 20 # kg embedding dim

    num_filters = 32 # num of cnn filters
    cnn_kernel_size = 3 # the kernel size

    fc_hidden_dim = 128  #

    # training
    batch_size = 32  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = '../Model/Transformer_log/'  # log directory

    # model
    class_num = 2  # num of kinds of rel
    maxlen = 50  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    min_cnt = 1  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


