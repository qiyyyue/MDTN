import codecs
from collections import Counter

from MultiChannelDTN.hyperparams import Hyperparams as hp

def len_cnt():
    len_cnt_list = []
    for label, content in [line.strip().split('<#>') for line in codecs.open(hp.train_path, 'r', 'utf-8').readlines() if line]:
        len_cnt_list.append(len(content.strip().replace('\t', ' ').split()))

    len_cnt_dict = Counter(len_cnt_list)
    for art_len, cnt in len_cnt_dict.most_common(len(len_cnt_dict)):
        print(art_len, cnt)

len_cnt()