import json
import numpy as np

class Dict(object):
    def __init__(self):
        self.word2id = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<&&&>': 4}
        self.id2word = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>', 4: '<&&&>'}
        self.frequency = {}

    def add(self, s):
        ids = []
        for w in s:
            if w in self.word2id:
                id = self.word2id[w]
                self.frequency[w] += 1
            else:
                id = len(self.word2id)
                self.word2id[w] = id
                self.id2word[id] = w
                self.frequency[w] = 1
            ids.append(id)
        return ids

    def transform(self, s):
        ids = []
        for w in s:
            if w in self.word2id:
                id = self.word2id[w]
            else:
                id = self.word2id['<UNK>']
            ids.append(id)
        return ids

    def prune(self, k):
        sorted_by_value = sorted(self.frequency.items(), key=lambda kv: -kv[1])
        newDict = Dict()
        newDict.add(list(zip(*sorted_by_value))[0][:k])
        return newDict

    def save(self, fout):
        return json.dump({'word2id': self.word2id, 'id2word': self.id2word}, fout, ensure_ascii=False)

    def load(self, fin):
        datas = json.load(fin)
        self.word2id = datas['word2id']
        self.id2word = datas['id2word']

    def __len__(self):
        return len(self.word2id)

def preprocess(data_kind, dir_path, ch_dict=None):
    print(data_kind)
    data_path = dir_path + data_kind + '.json'
    datas = json.load(open(data_path, 'r', encoding='utf-8'))

    for i in range(len(datas)):

        src_sentence = datas[i]['src']
        tgt_sentence = datas[i]['tgt']

        if ch_dict is not None:
            ch_dict.add(src_sentence.split())
            ch_dict.add(tgt_sentence.split())

        if i % 10000 == 0:
            print('{} sentences finished.'.format(i))

    if ch_dict is not None:
        ch_dict.save(open(dir_path + 'dict_whole.json', 'w', encoding='utf-8'))
        src_dict = ch_dict.prune(50000)
        src_dict.save(open(dir_path + 'dict_50000.json', 'w', encoding='utf-8'))

if __name__ == '__main__':
    preprocess('valid', './data/')
    preprocess('test', './data/')
    preprocess('train', './data/', Dict())

