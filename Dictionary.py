import os
import collections
import random

random.seed(210)


class LoadData(object):

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '__instance'):
            cls.__instance = super(LoadData, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        self.islower = True
        self.fine_tune = True
        self.shuffle = True
        self.all_file = []

        self.cwpath = os.getcwd()
        self.dtpath = self.cwpath + '/data/'
        self.file_name = os.listdir(self.dtpath)

    def form_data(self):

        for name in self.file_name:
            with open(self.dtpath + name, 'r') as f:
                per_file = []
                for idx, line in enumerate(f):
                    content = line.rstrip().split(' ||| ')
                    sentence = content[0].split()
                    label = content[1]
                    per_file.append(
                        {'example': sentence, 'label': label}
                    )

            if self.shuffle:
                random.shuffle(per_file)

            self.all_file.append({'name': name, 'data': per_file})


class Word(object):
    def __init__(self, data=None, min_freq=1,
                 islower=True, fine_tune=True):
        self.data = data
        self.vocab = collections.OrderedDict()
        self.label = collections.OrderedDict()
        self.islower = islower
        self.min_freq = min_freq
        self.fine_tune = fine_tune
        self.sentence_maxlen = 0
        self.vocab_num = 0
        self.label_num = 0

    def __word_process(self, sentence):
        if self.islower:
            if type(sentence) == list:
                for idx, word in enumerate(sentence):
                        sentence[idx] = word.lower()

    def __add_word(self, sentence, aim):
        if type(sentence) == list:
            for word in sentence:
                if word not in aim:
                    aim[word] = 1
                else:
                    aim[word] += 1
                # print(word, self.vocab[word])
        else:
            if sentence not in aim:
                aim[sentence] = 1
            else:
                aim[sentence] += 1

    def __read_word(self):

        '''
        if isinstance(file, str):
            print('Your input data file is ', file)
        else:
            raise Exception('Input the right data file!')
        '''

        '''
        if len(file) == 0:
            raise Exception("Data file is blank! " +
                            "Add the dataset in data file!")
        '''
        if self.fine_tune:
            self.data = [self.data[1]]        # Select the train data to build vocabulary.

        for per_file in self.data:
            content = per_file['data']
            for item in content:
                sentence = item['example']
                if len(sentence) > self.sentence_maxlen:
                    self.sentence_maxlen = len(sentence)
                label = item['label']
                self.__word_process(sentence)
                self.__add_word(sentence, self.vocab)
                self.__word_process(label)
                self.__add_word(label, self.label)

    def __order_dict(self, aim):
        sequence = sorted(aim.items(),
                          key=lambda x: x[1], reverse=True)
        new_dict = collections.OrderedDict()
        for item in sequence:
            new_dict[item[0]] = item[1]
        aim.clear()
        aim.update(new_dict)

    def build_dict(self):
        self.__read_word()

        self.__order_dict(self.vocab)
        self.vocab_num = len(self.vocab)

        self.__order_dict(self.label)
        self.label_num = len(self.label)

        print('Different vocabulary and label are collected.')


class WordTable(object):

    def __init__(self, vocab=None, max_len=None):
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
        self.stc_maxlen = max_len


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
        # self.stoi.update(self.table)
        #   This sentence lead to the difference of stoi and itos!!!
        for idx, item in enumerate(self.table):
            self.stoi.update(
                {item: idx + 2}
            )

    def build_table(self):
        self.__build_itos()
        self.__build_stoi()
        print('Word table is built.')

    def load_word2id(self, word):
        if word in self.stoi:
            return self.stoi[word]
        else:
            return self.unk_idx

    def load_label2id(self, label):
        #if label in self.stoi:
        #    return self.stoi[label]
        #else:
        #    return self.unk_idx

        label_dict = {'0': 0, '1': 1}
        try:
            return label_dict[label]
        except KeyError as e:
            return 2


if __name__ == '__main__':
    '''
    table = WordTable()
    print(table.traintable)
    print('aaa')
    '''

    '''
    word = Word(islower=True)
    word.build_dict()

    table = WordTable(word.vocab, word.sentence_maxlen)
    table.build_table()

    for i, j in zip(table.stoi, table.itos):
        print('%10s %20s %20s %20s' % (i, table.stoi[i], j, table.itos[j]))
    '''

    file = LoadData()
    file.form_data()

    word = Word(data=file.all_file, islower=True)
    word.build_dict()

    tabel = WordTable(word.vocab)
    tabel.build_table()
    print('999')




