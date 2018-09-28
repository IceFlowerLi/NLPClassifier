import random
import collections
import torch

from Dictionary import *

random.seed(0)
torch.manual_seed(0)


class BatchFeature(object):

    def __init__(self):
        self.inst = None
        self.batch = None
        self.batch_len = 0
        self.vector = []
        self.iterator = []
        self.data_iter = []


class Iterator(BatchFeature):

    def __init__(self, batch=None, data=None, config=None):
        super(BatchFeature, self).__init__()

        self.batch = batch
        self.data = data
        self.config = config

    def _word_id(self, inst, vocab):
        """
        :param inst:    Instance of Word.
        :param vocab:   Instance of the WordTable.
        :return:        None now.
        """
        line = inst.sentence_label
        line_wordid = collections.OrderedDict()
        line_labelid = collections.OrderedDict()

        for ldx, item in enumerate(line):
            label = item[-1]
            sentence = item[0:-1]

            for idx, word in enumerate(sentence):
                if idx < inst.sentence_maxlen:
                    line_wordid.update({idx:vocab.load_word2id(word)})
                else:
                    line_wordid.update({idx:vocab.pad_idx})

            line_labelid.update({ldx:vocab.load_label2id(label)})


    def create_iter(self):

        assert isinstance(self.data, list), "Error! Data input must be in list!"

        for idx in range(len(self.data)):
            pass

