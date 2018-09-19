import os
import collections


class Word:
    def __init__(self, datapath=None, min_freq=1,
                 islower=True, fine_tune=True):
        self.vocab = collections.OrderedDict()
        self.islower = islower
        self.min_freq = min_freq
        self.cwpath = os.getcwd()
        self.dtpath = self.cwpath + '/data/' if not datapath else datapath
        self.datafile = os.listdir(self.dtpath)
        self.fine_tune = fine_tune

    def __word_process(self, sentence):
        if self.islower:
            for idx, word in enumerate(sentence):
                    sentence[idx] = word.lower()
        return sentence

    def __add_word(self, sentence):
        for word in sentence:
            if word not in self.vocab:
                self.vocab[word] = 1
            else:
                self.vocab[word] += 1
            # print(word, self.vocab[word])

    def __read_word(self, file=None):
        if not file:
            file = self.datafile
        '''
        if isinstance(file, str):
            print('Your input data file is ', file)
        else:
            raise Exception('Input the right data file!')
        '''
        if len(file) == 0:
            raise Exception("Data file is blank! " +
                            "Add the dataset in data file!")
        if not self.fine_tune:
            file = [self.dtpath + 'train2.txt']

        for name in file:
            with open(self.dtpath + name, 'r') as f:
                for idx, line in enumerate(f):
                    content = line.split(' ||| ')
                    sentence = content[0].split()
                    label = content[1].split()
                    self.__word_process(sentence)
                    self.__add_word(sentence)

    def __order_dict(self):
        sequence = sorted(self.vocab.items(),
                          key=lambda x: x[1], reverse=True)
        new_dict = collections.OrderedDict()
        for item in sequence:
            new_dict[item[0]] = item[1]
        self.vocab = new_dict

    def build_dict(self, file=None):
        self.__read_word(file)
        self.__order_dict()


class WordTable:

    def __init__(self, vocab=None):
        self.pad_str = '<pad>'
        self.pad_idx = 0
        self.unk_str = '<unk>'
        self.unk_idx = 1
        self.itos = collections.OrderedDict()
        self.stoi = collections.OrderedDict()
        if not vocab:
            raise Exception('The dictionary is empty!')
        else:
            self.table = vocab

    def __build_itos(self):
        self.itos.update(
            {self.pad_idx: self.pad_str}
        )
        self.itos.update(
            {self.unk_idx: self.unk_str}
        )
        for idx, item in enumerate(self.table):
            self.itos.update(
                {idx + 2: item}
            )

    def __build_stoi(self):
        self.stoi.update({self.pad_str: self.pad_idx})
        self.stoi.update({self.unk_str: self.unk_idx})
        self.stoi.update(self.table)

    def build_table(self):
        self.__build_itos()
        self.__build_stoi()


if __name__ == '__main__':
    '''
    table = WordTable()
    print(table.traintable)
    print('aaa')
    '''
    word = Word(islower=False)
    word.build_dict()

#    for i in word.vocab:
#        print(i, word.vocab[i])

#    print(type(word.vocab))
#    print(word.vocab[','])

    table = WordTable(word.vocab)
    table.build_table()

    for i in table.itos:
        print("{0}: {1}".format(i, table.itos[i]))





