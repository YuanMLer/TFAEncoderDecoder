'''losses = (sum((y-pred)*(y-pred))/n ) /var )
(c)Minglei,Yuan
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

__all__= ['TotalVarMSEloss']



class TotalVarMSEloss(nn.Module):

    def __init__(self):
        super(TotalVarMSEloss, self).__init__()
        if torch.cuda.is_available():
            self.tvars = torch.FloatTensor(np.array([149.1082,   2.6687, 657.1512]))    #the golobal variabce
            self.tvars = self.tvars.cuda()
        else:
            self.tvars = torch.FloatTensor(np.array([149.1082, 2.6687, 657.1512]))  # the golobal variabce
    def forward(self, y, pred):
        """
        stand: loss = sum( pow( (y - y_pred) , 2 )/var( y ) ) / num
        We use :loss = sum( pow( (y - y_pred) , 2 )/(var( y ))) / num ;
        :param y: the real value [real_batch_size,sequence,dim]
        :param pred: the predict value [real_batch_size,sequence,dim]
        :param filepath: the file to get the y_var value
        :return: loss
        """
        theta = 50
        y_pow = torch.pow((y - pred), 2)
        loss1 = torch.sum(y_pow)
        loss2 = torch.sum(y_pow / self.tvars)
        loss = (loss1 + theta*loss2) / (y.size(0) * y.size(1) * y.size(2))
        return loss
