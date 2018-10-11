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

    def __init__(self, batch=None, batch_len=None,
                 data=None, vocab=None, config=None):
        super(BatchFeature, self).__init__()

        self.batch = batch

        assert batch_len < len(data), "ERROR! Batch size over the data length!"
        self.batch_len = batch_len

        self.data = [data] if type(data) != list else data
        self.vocab = vocab
        self.config = config

    def _word_id(self, inst, vocab):
        """
        :param inst:    Instance of Word.
        :param vocab:   Instance of the WordTable.
        :return:        None now.
        """
        # line = inst.sentence_label
        line = inst
        wordid = []
        labelid = []

        for item in line:
            label = line[item][1]
            sentence = line[item][0].split()
            per_stc = []

            for idx, word in enumerate(sentence):
                per_stc.append(vocab.load_word2id(word.lower()))

            # Keep the same length of sentence.
            # word_gap = self.batch.sentence_maxlen - len(sentence)
            # if word_gap > 0:
            #     per_stc = per_stc + [vocab.pad_idx for i in range(word_gap)]

            wordid.append(per_stc)
            labelid.append(vocab.load_label2id(label))

        return wordid, labelid

    def create_perbatch(self, inst, vocab):
        perbatch_word = []
        perbatch_label = []

        word_vct, label_vct = self._word_id(inst, vocab)

        for group in range(len(word_vct) // self.batch_len + 1):

            start, end = group * self.batch_len, (group + 1) * self.batch_len

            if end < len(word_vct):
                perbatch_word.append(word_vct[start:end])
                perbatch_label.append(label_vct[start:end])
            else:
                supply_word = []
                supply_label = []
                extra = end - len(word_vct)

                for i in range(extra):
                    supply_word.append(
                        [vocab.pad_idx for i in range(self.batch.sentence_maxlen)]
                    )
                    supply_label.append(vocab.load_label2id('Unkown'))

                perbatch_word.append(word_vct[start:] + supply_word)
                perbatch_label.append(label_vct[start:] + supply_label)

        return perbatch_word, perbatch_label

#    def create_batch(self, inst, vocab, batchid):
#
#        # word_vct = torch.zeros([self.batch_len, inst.sentence_maxlen],
#        #                        dtype=torch.long)
#        # label_vct = torch.zeros(self.batch_len, dtype=torch.long)
#
#        batch_word = []
#        batch_label = []
#
#        for item in batchid:
#            batch_word.append(item)
#            batch_label.append(item)
#            pass

    def create_iter(self):

        # assert isinstance(self.data, list), "Error! Data input must be in list!"

        data_iter = []
        label_iter = []
        for sgl_data in self.data:
            dt_it, lb_it = self.create_perbatch(sgl_data, self.vocab)
            data_iter.append(dt_it)
            label_iter.append(lb_it)

        return data_iter, label_iter


if __name__ == '__main__':
    word = Word(islower=True)
    word.build_dict()

    table = WordTable(word.vocab, word.sentence_maxlen)
    table.build_table()

    batch_iter = Iterator(batch=word, batch_len=500,
                          data=word.sentence_label, vocab=table)
    data_iter, label_iter = batch_iter.create_iter()
    print('')





