# -*- coding: utf-8 -*-
from __future__ import print_function
from MultiChannelDTN.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
from Model.kg_embedding_model.triple2vec import triple2vec
from sklearn.utils import shuffle

def load_vocab():
    vocab = [line.split()[0] for line in codecs.open(hp.vocab_path, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_label2id():
    relid2target_dict = {'fake': 0, 'true': 1}
    return relid2target_dict

def triple_kg_embd(content, kg_embedding_model):
    content = content.split('\t')
    re_list = []
    for i in range(len(content)):
        if i%3 == 1:
            re_list.append(list(kg_embedding_model.relation2vec(content[i].strip())))
        else:
            re_list.append(list(kg_embedding_model.entity2vec(content[i].strip())))
    re_list.append([0]*20)
    return re_list

def create_data(sentences, targets, kg_embedding_model):

    word2id, id2word = load_vocab()
    # Index
    trans_x_list, kg_embd_x_list, y_list, Sentences, Targets = [], [], [], [], []

    print('create data')
    for i, (sentence, target) in enumerate(zip(sentences, targets)):
        # print('processing No.{}'.format(i))
        trans_x = [word2id.get(word, 1) for word in (sentence.strip().replace('\t', ' ') + u" </S>").split()] # 1: OOV, </S>: End of Text
        kg_embd_x = []
        for content in sentence.strip().split('<END>'):
            if content:
                kg_embd_x += triple_kg_embd(content.strip(), kg_embedding_model)
        y = [0]*hp.class_num
        y[target] = 1

        if len(trans_x) > hp.maxlen:
            trans_x = trans_x[:hp.maxlen]
        if len(kg_embd_x) > hp.maxlen:
            kg_embd_x = kg_embd_x[:hp.maxlen]
        trans_x_list.append(np.array(trans_x))
        kg_embd_x_list.append(np.array(kg_embd_x))
        y_list.append(np.array(y))
        Sentences.append(sentence)
        Targets.append(target)
    print('create finished')

    # Pad      
    trans_X = np.zeros([len(trans_x_list), hp.maxlen], np.int32)
    kg_embd_X = np.zeros([len(trans_x_list), hp.maxlen, hp.kg_embd_dim], np.float32)
    Y = np.zeros([len(y_list), hp.class_num], np.int32)
    for i, (trans_x, kg_embd_x, y) in enumerate(zip(trans_x_list, kg_embd_x_list, y_list)):
        trans_X[i] = np.lib.pad(trans_x, [0, hp.maxlen-len(trans_x)], 'constant', constant_values=(0, 0))
        #if hp.use_kg_embd:
        kg_embd_X[i] = np.lib.pad(kg_embd_x, ((0, (hp.maxlen - len(kg_embd_x))), (0, 0)), 'constant',
                                      constant_values=0)
        Y[i] = y
    
    return trans_X, kg_embd_X, Y, Sentences, Targets

def load_train_data():
    kg_embedding_model = triple2vec(20, "../Model/kg_embedding_model/dbpedia_model/e_tr/")
    label2id = load_label2id()

    train_triples = []
    train_labels = []
    for label, content in [line.strip().split('<#>') for line in codecs.open(hp.train_path, 'r', 'utf-8').readlines() if line]:
        train_triples.append(content.strip())
        train_labels.append(label2id[label])

    trans_X, kg_embd_X, Y, Sources, Targets = create_data(train_triples, train_labels, kg_embedding_model)
    return trans_X, kg_embd_X, Y

def load_val_data():
    kg_embedding_model = triple2vec(20, "../Model/kg_embedding_model/dbpedia_model/e_tr/")
    label2id = load_label2id()

    train_triples = []
    train_labels = []
    for label, content in [line.strip().split('<#>') for line in codecs.open(hp.val_path, 'r', 'utf-8').readlines() if
                           line]:
        train_triples.append(content.strip())
        train_labels.append(label2id[label])

    # print('len', len(train_sentences), len(train_targets))
    trans_X, kg_embd_X, Y, Sources, Targets = create_data(train_triples, train_labels, kg_embedding_model)
    return trans_X, kg_embd_X, Y

def load_test_data():
    kg_embedding_model = triple2vec(20, "../Model/kg_embedding_model/dbpedia_model/e_tr/")
    label2id = load_label2id()

    train_triples = []
    train_labels = []
    for label, content in [line.strip().split('<#>') for line in codecs.open(hp.test_path, 'r', 'utf-8').readlines() if
                           line]:
        train_triples.append(content.strip())
        train_labels.append(label2id[label])

    # print('len', len(train_sentences), len(train_targets))
    trans_X, kg_embd_X, Y, Sources, Targets = create_data(train_triples, train_labels, kg_embedding_model)
    return trans_X, kg_embd_X, Y

def get_batch_data():

    trans_X, kg_X, Y = load_train_data()
    # calc total batch count
    num_batch = len(trans_X) // hp.batch_size

    indices = np.random.permutation(np.arange(len(trans_X)))
    trans_X = trans_X[indices]
    kg_X = kg_X[indices]
    Y = Y[indices]

    for i in range(num_batch):
        batch_trans_x = trans_X[i*hp.batch_size: (i + 1)*hp.batch_size]
        batch_kg_x = kg_X[i*hp.batch_size: (i + 1)*hp.batch_size]
        batch_y = Y[i*hp.batch_size: (i + 1)*hp.batch_size]
        yield  batch_trans_x, batch_kg_x, batch_y


# for x, y in get_batch_data():
#     print(x.shape, y.shape)
#     print(x)
#     print(y)
#     print('------------------------------')


# t2v = triple2vec(20, "../Model/kg_embedding_model/dbpedia_model/e_tr/")
# trans_X, kg_embd_X, Y = load_train_data(t2v)
# print(trans_X.shape)
# print(kg_embd_X.shape)
# print(Y.shape)

# a = np.array([[1,2], [3,4]])
# print(np.lib.pad(a, ((1,3),(0,0)),'constant', constant_values=0))

# def length(sequence):
#   used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
#   length = tf.reduce_sum(used, 1)
#   length = tf.cast(length, tf.int32)
#   return length
#
# a = [i for i in range(5000)]
# a = np.reshape(np.array(a), [-1, 50, 20])
# a[0:4, 0:2, :] = 0
# a[4, 0:9, 1:] = 0
# print(a)
# ss_len = length(a)
# session = tf.Session()
# s_len = session.run(ss_len)
# print(s_len.shape)
# print(s_len)