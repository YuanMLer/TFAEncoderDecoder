# -*- encoding:utf-8 -*-
# predict the weather information use the lstm model according to the history information
import torch
import torch.nn as nn
from torch.autograd import Variable


__all__ = [
    'purelstm',
    'PureLSTM'
]

use_cuda = torch.cuda.is_available()

class PureLSTM(nn.Module):

    def __init__(self, args):
        super(PureLSTM, self).__init__()
        self.args = args
        # batch normal for the input data
        self.bnForInput = nn.BatchNorm1d(self.args.seq_len)
        # rnnGetC get the C value as the input of rnnGetPred
        self.rnnGetC = nn.LSTM(input_size=args.natural_encoder_in_dim,
                          hidden_size=args.encoder_hidden_dim,
                          num_layers=args.encoder_num_layer,
                          bidirectional=args.bidirectional,
                          batch_first=args.batch_first)

        bidirection = 2 if args.bidirectional else 1

        # rnnGetPred get the observe value use the super calculated value
        self.rnnGetPred = nn.LSTM(input_size=self.args.encoder_hidden_dim * bidirection,
                                  hidden_size=self.args.encoder_hidden_dim,
                                  num_layers=self.args.encoder_num_layer,
                                  bidirectional=self.args.bidirectional,
                                  batch_first=self.args.batch_first)
        self.sequenceForOut = nn.Sequential(nn.Linear(self.args.encoder_hidden_dim * bidirection,
                                                      len(self.args.pred_index)))

    # def forward(self, inputs,pred_len):
    def forward(self,inputs,predict_M, getAttention=False):
        """
        do the LSTM process
        :param inputs: the input sequcen
        :param pred_len: the length of the predict value
        :return: outputs:the ouput sequence
        """
        assert len(inputs.size()) == 3, '[LSTM]: input dimension must be of length 3 i.e. [M*S*D]'

        # do do the first RNN
        if self.args.bidirectional:
            bi_num = 2
        else:
            bi_num = 1

        r_batch_size = inputs.size(0)       #the real batchsize
        # print("inputs.size() = {}".format(inputs.size()))

        inputs = self.bnForInput(inputs)

        if self.args.batch_first:
            h = [Variable(torch.randn(r_batch_size, bi_num * self.args.encoder_num_layer,
                                      self.args.encoder_hidden_dim)),
                 Variable(torch.randn(r_batch_size, bi_num * self.args.encoder_num_layer,
                                      self.args.encoder_hidden_dim))]
        else:
            h = [Variable(torch.randn(bi_num * self.args.encoder_num_layer, r_batch_size,
                                     self.args.encoder_hidden_dim)),
                 Variable(torch.randn(bi_num * self.args.encoder_num_layer, r_batch_size,
                                      self.args.encoder_hidden_dim))]
            inputs = inputs.permute(1, 0, 2)
        if torch.cuda.is_available():
            h = [h[0].cuda(), h[1].cuda()]

        out, h = self.rnnGetC(inputs,h)

        if self.args.batch_first:
            last_out = out[:,-1,:]
            last_out = last_out.unsqueeze(1)
        else:
            last_out = out[-1,:,:]
            last_out = last_out.unsqueeze(0)

        #do the seconde RNN
        outs = []
        temp_out = last_out
        for i in range(self.args.pred_len):
            temp_out, h = self.rnnGetPred(temp_out,h)
            outs.append(temp_out)
        outs = torch.stack(outs, dim=0)
        outs = outs.squeeze(1)

        # get result
        outs = self.sequenceForOut(outs)

        if not self.args.batch_first:
            outs = outs.permute(1,0,2)
        return outs


def purelstm(**kwargs):
    r"""pure lstm, model architecture
    :return:  pure lstm model
    """
    model = PureLSTM(**kwargs)
    return model
