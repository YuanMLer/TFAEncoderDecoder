# -*- encoding:utf-8 -*-
# predict the weather information use the lstm model according to the history information
# use Encoder and Decoder and Temporal Attention and super computer value
# (We only use encoder last Hidden message and supper computer value to get the attention weight)
# key words: (hidden + super calculate value) as the attention function input and decoder input
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    'tsaendecoder'
]

use_cuda = torch.cuda.is_available()

class TemporalSuperCalAttentionEncoderDecoder(nn.Module):

    def __init__(self, args):
        super(TemporalSuperCalAttentionEncoderDecoder, self).__init__()
        self.args = args
        self.bn = nn.BatchNorm1d(self.args.seq_len)
        # rnnGetC get the C value as the input of rnnGetPred
        self.encoder = nn.LSTM(input_size=args.natural_encoder_in_dim,
                               hidden_size=args.encoder_hidden_dim,
                               num_layers=args.encoder_num_layer,
                               bidirectional=args.bidirectional,
                               batch_first=args.batch_first)

        self.bi_num = 2 if args.bidirectional else 1
        # Sequence for the decoder attention
        self.sequenceForAttention = nn.Sequential(nn.Linear(args.encoder_hidden_dim * args.encoder_num_layer *
                                                                   self.bi_num + len(args.supercomput_index),
                                                                   args.seq_len))
        self.sequenceForDecoder = nn.Sequential(nn.Linear(args.encoder_hidden_dim + len(args.supercomput_index),
                                                            args.decoder_hidden_dim))
        # rnnGetPred get the observe value use the super calculated value
        self.decoder = nn.LSTM(input_size=args.encoder_hidden_dim * self.bi_num,
                               hidden_size=args.decoder_hidden_dim,
                               num_layers=args.decoder_num_layer,
                               bidirectional=args.bidirectional,
                               batch_first=args.batch_first)

        self.sequenceForOut = nn.Sequential(nn.Linear(args.decoder_hidden_dim * self.bi_num,len(args.pred_index)))

    def forward(self, inputs, predict_M, getAttention=False):
        """
        do the LSTM process
        :param inputs: the input sequcen
        :param pred_len: the length of the predict value
        :return: outputs:the ouput sequence
        """
        assert len(inputs.size()) == 3, '[LSTM]: input dimension must be of length 3 i.e. [M*S*D]'
        # encoder
        inputs = self.bn(inputs)
        inputs, h, r_batch_size = self._prepare_input_h(inputs)
        c, h = self.encoder(inputs, h)
        # attention and decoder
        outs = []
        predict_M, _, _ = self._prepare_input_h(predict_M)
        # this must be in the outside of the for loop
        if not self.args.batch_first:
            c = c.permute(1, 0, 2)  # c batch first
        h_ = h  # we use the h from encoder h_ = [(args.encoder_num_layer*binum,batch_size,r_batch_size,encoder_hidden_dim),
        #                                       (args.encoder_num_layer*binum,batch_size,r_batch_size,encoder_hidden_dim)]

        if getAttention:
            attention = []

        for i in range(self.args.pred_len):
            # attention
            h_temp = h_[0]
            pred = predict_M[:, i, :] if self.args.batch_first else predict_M[i, :, :]
            h_temp = h_temp.contiguous().view(r_batch_size,-1)
            (h_temp, pred) = (torch.unsqueeze(h_temp, 1),torch.unsqueeze(pred, 1)) \
                if self.args.batch_first else (torch.unsqueeze(h_temp, 0),torch.unsqueeze(pred, 0))
            weights = F.softmax(self.sequenceForAttention(torch.cat((h_temp, pred), dim=-1)), dim=-1)

            if not self.args.batch_first:
                weights = weights.permute(1, 0, 2)  # weight batch first

            if getAttention:
                temp_weights = weights.squeeze()
                temp_weights = temp_weights.cpu().detach().numpy()
                attention.append(temp_weights.tolist())

            out = torch.bmm(weights, c)  # [r_batch_size,1,encoder_hidden_dim]
            if not self.args.batch_first:
                out = out.permute(1, 0, 2)
            # decoder
            # print("out.size = {}".format(out.size()))
            # print("pred.size = {}".format(pred.size()))
            out = self.sequenceForDecoder(torch.cat((out, pred),dim=-1))
            out, h_ = self.decoder(out, h_)
            outs.append(out)
        outs = torch.stack(outs, dim=0)
        outs = torch.squeeze(outs, dim=2) if self.args.batch_first else torch.squeeze(outs, dim=1)
        outs = self.sequenceForOut(outs)

        if not self.args.batch_first:
            outs = outs.permute(1, 0, 2)

        if getAttention:
            return outs, attention
        else:
            return outs

    def _prepare_input_h(self, inputs):
        r_batch_size = inputs.size(0)  # the real batchsize
        if self.args.batch_first:
            h = [Variable(torch.randn(r_batch_size, self.bi_num * self.args.encoder_num_layer,
                                      self.args.encoder_hidden_dim)),
                 Variable(torch.randn(r_batch_size, self.bi_num * self.args.encoder_num_layer,
                                      self.args.encoder_hidden_dim))]
        else:
            h = [Variable(torch.randn(self.bi_num * self.args.encoder_num_layer, r_batch_size,
                                      self.args.encoder_hidden_dim)),
                 Variable(torch.randn(self.bi_num * self.args.encoder_num_layer, r_batch_size,
                                      self.args.encoder_hidden_dim))]
        if torch.cuda.is_available():
            h = [h[0].cuda(), h[1].cuda()]
        if not self.args.batch_first:
            inputs = inputs.permute(1, 0, 2)
        return inputs, h, r_batch_size


def tsaendecoder(**kwargs):
    r"""tsaendecoder, model architecture
    :return:  pure lstm model
    """
    model = TemporalSuperCalAttentionEncoderDecoder(**kwargs)
    return model
