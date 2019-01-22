'''losses = sum((y-pred)*(y-pred))/n
(c)Minglei,Yuan
'''
import torch
import torch.nn as nn


# torch.random.manual_seed(1)
# y = torch.randn(2,2,2)
# pred = torch.randn(2,2,2)
#
# print("y = {}".format(y))
# print("pred = {}".format(pred))

__all__= ['mlMSEloss']


class mlMSEloss(nn.Module):

    def __init__(self):
        super(mlMSEloss, self).__init__()

    def forward(self, y, pred):
        """

        We use :loss = sum( pow( (y - y_pred) , 2 )) / num ;
        if batch_size =1 we use MSELoss
        :param y: the real value [real_batch_size,sequence,dim]
        :param pred: the predict value [real_batch_size,sequence,dim]
        :return: loss
        """

        y_pow = torch.pow((y - pred), 2)
        # print("y_pow = {}".format(y_pow))
        loss = torch.sum(y_pow) / (y.size(0) * y.size(1) * y.size(2))
        return loss


# def main():
#     M = mlMSEloss()
#     loss = M(y, pred)
#     print("mlMSE = {}".format(loss))
#     criterion = torch.nn.MSELoss()
#     loss = criterion(pred, y)
#     print("MSELoss = {}".format(loss))
#
#
# if __name__ == '__main__':
#     main()
