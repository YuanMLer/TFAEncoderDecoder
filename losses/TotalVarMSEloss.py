'''losses = (sum((y-pred)*(y-pred))/n ) /var )
(c)Minglei,Yuan
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# torch.random.manual_seed(1)
# y = torch.randn(2,2,3)
# pred = torch.randn(2,2,3)
# print("y = {}".format(y))
# print("pred = {}".format(pred))


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


# threshold of weather variant
# threshold = {'psfc_M': [850, 1100],
#              't2m_M': [-40, 55],
#              'q2m_M': [0, 30],
#              'rh2m_M': [0, 100],
#              'w10m_M': [0, 30],
#              'd10m_M': [0, 360],
#              'u10m_M': [-30, 30],
#              'v10m_M': [-30, 30],
#              'SWD_M': [0, 1500],
#              'GLW_M': [0, 800],
#              'HFX_M': [-400, 1000],
#              'LH_M': [-100, 1000],
#              'RAIN_M': [0, 400],
#              'PBLH_M': [0, 6000],
#              'TC975_M': [-50, 45],
#              'TC925_M': [-50, 45],
#              'TC850_M': [-55, 40],
#              'TC700_M': [-60, 35],
#              'TC500_M': [-70, 30],
#              'wspd975_M': [0, 60],
#              'wspd925_M': [0, 70],
#              'wspd850_M': [0, 80],
#              'wspd700_M': [0, 90],
#              'wspd500_M': [0, 100],
#              'Q975_M': [0, 30],
#              'Q925_M': [0, 30],
#              'Q850_M': [0, 30],
#              'Q700_M': [0, 25],
#              'Q500_M': [0, 25],
#              'psur_obs': [850, 1100],
#              't2m_obs': [-40, 55],
#              'q2m_obs': [0, 30],
#              'rh2m_obs': [0, 100],
#              'w10m_obs': [0, 30],
#              'd10m_obs': [0, 360],
#              'u10m_obs': [-30, 30],
#              'v10m_obs': [-30, 30],
#              'RAIN_obs': [0, 400]
#              }
# def getvar(self, "F:\\workspace\\Bidecoder\\data\\csvfiles\\train",threshold):
#     indexes = [30, 34, 32]
#     temp_datas = []
#     for filepath in os.listdir(fileroot):
#
#         filepath = os.path.join(fileroot, filepath)
#         if filepath.endswith(".csv"):
#             data = pd.read_csv(filepath, header=0, usecols=None, index_col=None)
#             del_index = set([])
#             nan_index = np.where(np.isnan(data))[0]
#             for i_temp in nan_index:
#                 del_index.add(i_temp)
#             for col in threshold:
#                 t = set(data[data[col] < threshold[col][0]].index)
#                 del_index |= t
#                 t = set(data[data[col] > threshold[col][1]].index)
#                 del_index |= t
#                 t = set(data[data[col] == -9999].index)
#                 del_index |= t
#             del_index = sorted(del_index)
#             data = pd.read_csv(filepath, header=0, usecols=indexes, index_col=None)
#             data = data.drop(del_index)
#             temp = data.values
#             temp_datas.append(temp)
#
#     temp_datas = np.array(temp_datas)
#     datas = []
#     for i in range(10):
#         for t in temp_datas[i]:
#             datas.append(t)
#
#     temp_var = np.var(datas, axis=0)
#     temp_var = torch.FloatTensor(temp_var)
#     print(temp_var)
#     return temp_var
#
# def main():
#     M = TotalVarMSEloss()
#     filepath = "testA_27_sta0-24.csv"
#     loss = M(y, pred,filepath)
#     # print("TotalVarMSEloss = {}".format(loss))
#     criterion = torch.nn.MSELoss()
#     loss = criterion(pred, y)
#     # print("MSEloss = {}".format(loss))
#
#
# if __name__ == '__main__':
#     main()
