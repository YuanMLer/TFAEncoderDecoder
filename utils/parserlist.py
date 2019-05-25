#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os


__all__ = ['ParserList']
# return:
#   args:parse objects
#   state:a map include all parses
def ParserList():
    parser = argparse.ArgumentParser(description="Weather Prediction")
    #Dataset
    parser.add_argument('-d','--dataset',default='WhetherPred1',type=str)
    parser.add_argument('-j','--workers',default=0,type=int,metavar='N',
                        help='number of dataset loading workers(default:4)')
    #Path
    parser.add_argument('--basedir', default=os.path.join('/home','yml','workspace','Bidecoder','data'),
                        type=str, metavar='PATH', help='base path of dataset')

    # parser.add_argument('--basedir', default=os.path.join('F:\workspace\WeatherPredictAIChallange\data'))
    parser.add_argument('--traindir', default=os.path.join('csvfiles','train'),
                        type=str, metavar='PATH',help='path of train dataset')
    parser.add_argument('--validationdir', default=os.path.join('csvfiles', 'validation'),
                        type=str, metavar='PATH', help='path of validation dataset')
    parser.add_argument('--testdir', default=os.path.join('csvfiles','testB'),
                        type=str, metavar='PATH',help='path of test dataset')

    parser.add_argument('--m-traindir-base',
                        default=os.path.join('csvfiles','singleStationDataTrain'),
                        type=str, metavar='PATH',
                        help='path of single model train dataset')
    parser.add_argument('--m-validationdir-base',
                        default=os.path.join('csvfiles', 'singleStationDataValidation'),
                        type=str, metavar='PATH',
                        help='path of single model validation dataset')
    parser.add_argument('--m-testdir-base',
                        default=os.path.join('csvfiles','singleStationDataValidation'),
                        type=str, metavar='PATH',
                        help='path of single model test dataset')
    #Optimization option
    parser.add_argument('--epoches',default=200,type=int, help='Number of total epoches to run')
    parser.add_argument('--refine-epoches', default=200, type=int, help='Number of refine epoches to run')
    parser.add_argument('--adjust-per-epochs', default=30, type=int, help='Number for epoches to reduce')
    parser.add_argument('--start-epoch',default=0,type=int,metavar='N',
                        help='manual epoch number(useful on restarts)')
    parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=None,
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint',
                        type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume',default=True,type=bool,metavar='PATH',
                        help='path to lastest checkpoint(default:none)')
    parser.add_argument('--resume-file', default='checkpoint.pth.tar', type=str, metavar='PATH',
                        help='file name of latest checkpoint file(default: none)')

    parser.add_argument('--train-single', default=True, type=bool,
                        help='Wheather train single station model or not')
    parser.add_argument('--test', default=False, type=bool,
                        help='Wheather only test the modle or not ')

    # Architecture
    import sys
    sys.path.append("..")
    import models

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser.add_argument('--arch', '-a', metavar='ARCH', default='puregru',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: AttribteTemporalSuperCalAttentionEncoderDecoder)')

    parser.add_argument('--criterion', metavar='LOSS', default='MSE',
                        choices=['MSE','RMSE','MAE','TotalVarMSEloss'],
                        help='loss funtion (default: Dofault loss is MSE)')

    parser.add_argument('--optimizer', metavar='OPTIM', default='SGD',
                        choices=['SGD','Adam'],
                        help='optimizer function(default: Default Optimizer function is SGD)')

    #LSTM
    parser.add_argument('--batch-first', default=False, type=bool,help='is batch in the first sequence')
    parser.add_argument('--bias', default=True, type=bool, help='use the bias or not')
    parser.add_argument('--bidirectional', default=False, type=bool, help='use the bias or not')
    parser.add_argument('--natural-encoder-in-dim',type=int,default=38,help="the natural input dim of the encoder")
    parser.add_argument('--encoder-in-dim',type=int,default=64,help="the input dimension of encoder")
    parser.add_argument('--encoder-hidden-dim',type=int,default=64,help="the hidden dimension of the rnn encoder models")
    parser.add_argument('--encoder-num-layer',type=int,default=2,help="the rnn layers of the encoder model")
    parser.add_argument('--natural-decoder-in-dim',type=int,default=29,help="the natural input dim of the decoder")
    # parser.add_argument('--decoder-in-dim',type=int, default=64,help="the input dimension of decoder")
    parser.add_argument('--decoder-hidden-dim', type=int, default=64, help="the hidden dimension of the rnn decoder models")
    parser.add_argument('--decoder-num-layer', type=int, default=2, help="the rnn layers of the decoder model")
    parser.add_argument('--decoder-out-dim',type=int,default=3,help="the output dimension of decoder")
    parser.add_argument('--seq-len',type=int,default=48,help="the history sequence length")
    parser.add_argument('--pred-len', type=int, default=2,help="the predict sequence length")
    parser.add_argument('--supercomput-index', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                 24, 25, 26, 27, 28],help="")
    parser.add_argument('--observe-index', type=int, nargs='+',
                        default=[29, 30, 31, 32, 33, 34, 35, 36, 37],help="")
    parser.add_argument('--pred-index', type=int, nargs='+',
                        default=[30,32,34])

    # parser.add_argument('--predict-dims', type=int, default=3)

    # Miscs
    parser.add_argument('--manual-seed', type=int, default=100, help='manual seed')
    parser.add_argument('--day-points', type=int, default=24, help='the points of a days')
    parser.add_argument('--rewrite-dataset-files',type=bool,default=False,help='rewrite the dataset files or not')
    #Device options
    parser.add_argument('--gpu-id', default='4', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    state = {k:v for k,v in args._get_kwargs()}
    new_state = {}
    for k in sorted(state.keys()):
        new_state[k] = state[k]
    return (args,new_state)

if __name__ == '__main__':
    ParserList()

