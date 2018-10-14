import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D, max_norm=10, padding_idx=1)

        if args.pretrain_embedding:
            idx_dict = self.idxtowrds(args.itos)
            self.load_embedding(idx_dict)

            self.embed.weight.reqiure_grad = True

        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        # self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        x = self.dropout(x)  # (N, len(Ks)*Co)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

    def load_embedding(self, idx_dict):
        print('Load pretrained embeding...')
        #with open('extract_googleNews_embed_sst.txt', 'r') as f:
        with open('glove.sentiment.conj.pretrained.txt', 'r') as f:
            first_line = next(f).rstrip().split()
            word_num = int(first_line[0])
            #word_dim = int(first_line[1])
            for idx, line in enumerate(f):
                content = line.rstrip().split()
                word = content.pop(0).lower()
                vector = [float(elem) for elem in content]
                if len(vector) > 300:
                    print('The large array in %d line.' % (idx + 1))
                    continue
                float_tensor = torch.tensor(vector, dtype=torch.float)
                idx = idx_dict.get(word)
                if idx:
                    self.embed.weight.data[idx].copy_(float_tensor)

    def idxtowrds(self, word_list):
        new_dict = dict()
        for idx, word in enumerate(word_list):
            new_dict[word] = idx
        return new_dict




