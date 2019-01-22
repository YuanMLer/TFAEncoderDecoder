'''losses = (sum((y-pred)*(y-pred))/n ) /(sum((y-ave_y)*(y-ave_y))/(n-1) )
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

__all__= ['thetaMSEloss']


class thetaMSEloss(nn.Module):

    def __init__(self):
        super(thetaMSEloss, self).__init__()

    def forward(self, y, pred, filepath):
        """
        stand: loss = sum( pow( (y - y_pred) , 2 )/var( y ) ) / num
        We use :loss = sum( pow( (y - y_pred) , 2 )/(var( y ))) / num ;
        if batch_size =1 we use MSELoss
        :param y: the real value [real_batch_size,sequence,dim]
        :param pred: the predict value [real_batch_size,sequence,dim]
        :param filepath: the file to get the y_var value
        :return: loss
        """

        if (y.size(0) == 1):
            criterion = torch.nn.MSELoss()
            loss = criterion(y, pred)
        else:
            y_var = torch.var(y, dim=0, keepdim=True, unbiased=True)
            # print("y_var = {}".format(y_var))
            y_pow = torch.pow((y - pred), 2)
            # print("y_pow = {}".format(y_pow))
            loss_sum = torch.div(y_pow, y_var)
            loss = torch.sum(loss_sum) / (y.size(0) * y.size(1) * y.size(2))
        return loss


# def main():
#     M = thetaMSEloss()
#     loss = M(y, pred)
#     print("ThetaMSE = {}".format(loss))
#     criterion = torch.nn.MSELoss()
#     loss = criterion(pred, y)
#     print("MSELoss = {}".format(loss))
#
#
# if __name__ == '__main__':
#     main()
