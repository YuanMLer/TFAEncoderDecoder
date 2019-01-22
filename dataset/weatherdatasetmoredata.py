# -*- encoding:utf-8 -*-
import os
import pickle as pk
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.autograd import Variable

__all__ = ['WeatherDatasetMoreData']

# threshold of weather variant
threshold = {'psfc_M': [850, 1100],
             't2m_M': [-40, 55],
             'q2m_M': [0, 30],
             'rh2m_M': [0, 100],
             'w10m_M': [0, 30],
             'd10m_M': [0, 360],
             'u10m_M': [-30, 30],
             'v10m_M': [-30, 30],
             'SWD_M': [0, 1500],
             'GLW_M': [0, 800],
             'HFX_M': [-400, 1000],
             'LH_M': [-100, 1000],
             'RAIN_M': [0, 400],
             'PBLH_M': [0, 6000],
             'TC975_M': [-50, 45],
             'TC925_M': [-50, 45],
             'TC850_M': [-55, 40],
             'TC700_M': [-60, 35],
             'TC500_M': [-70, 30],
             'wspd975_M': [0, 60],
             'wspd925_M': [0, 70],
             'wspd850_M': [0, 80],
             'wspd700_M': [0, 90],
             'wspd500_M': [0, 100],
             'Q975_M': [0, 30],
             'Q925_M': [0, 30],
             'Q850_M': [0, 30],
             'Q700_M': [0, 25],
             'Q500_M': [0, 25],
             'psur_obs': [850, 1100],
             't2m_obs': [-40, 55],
             'q2m_obs': [0, 30],
             'rh2m_obs': [0, 100],
             'w10m_obs': [0, 30],
             'd10m_obs': [0, 360],
             'u10m_obs': [-30, 30],
             'v10m_obs': [-30, 30],
             'RAIN_obs': [0, 400]
             }



# weather dataloader
class WeatherDatasetMoreData(Dataset):
    def __init__(self,root_dir,args,threshold=threshold):
        """
        :param root_dir: the root path of the weather data csv files
        :param args: the Parameters set
        :param device: to run in device
        :param threshold: the threshold of all data
        """""
        super(WeatherDatasetMoreData, self).__init__()
        self.root_dir = root_dir
        self.args = args
        self.threshold = threshold
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.supercomput_index = args.supercomput_index
        self.observe_index = args.observe_index
        self.pred_index = args.pred_index
        self.all_index = self.supercomput_index + self.observe_index
        self.pickle_file = os.path.join(self.root_dir,"data_{}_{}_{}.pkl".format(self.seq_len,self.pred_len,len(args.pred_index)))
        self.datas, self.seqs, self.length = self._pre_data(self.seq_len,self.pred_len,self.threshold)

    def __len__(self):
        # print("len = {}".format(self.length))
        return self.length

    def __getitem__(self, item):
        """
        get item in the num sequence.item indicated the number sequence of the datas
        :param item: the sequence of the id
        :return:
            history_msg:the sequence of the history in item order
            predict_M: the sequence of the predicted supercomputer result in item order
            predict_obs: the sequence of the observer 
        """""
        history_msg = []
        predict_M = []
        predict_obs = []

        for i in range(len(self.seqs)):
            if item in range(self.seqs[i][0], self.seqs[i][1]+1):
                bias_len = item - self.seqs[i][0]  # the start bias from the seqs[i][0]
                for temp_idx in range(bias_len, bias_len + self.seq_len):
                    history_msg.append(self.datas[i][temp_idx][self.all_index])  # history data from the datas
                for temp_idx in range(bias_len + self.seq_len, bias_len + self.seq_len + self.pred_len):
                    predict_M.append(self.datas[i][temp_idx][self.supercomput_index])
                    predict_obs.append(self.datas[i][temp_idx][self.pred_index])
                break

        history_msg = torch.from_numpy(np.array(history_msg))
        predict_M = torch.from_numpy(np.array(predict_M))
        predict_obs = torch.from_numpy(np.array(predict_obs))

        history_msg, predict_M, predict_obs = history_msg.float(), predict_M.float(), predict_obs.float()

        history_msg = Variable(history_msg)
        predict_M = Variable(predict_M)
        predict_obs = Variable(predict_obs)
        return history_msg, predict_M, predict_obs


    # preprocessing data
    def _pre_data(self,seq_len,pred_len,threshold):
        """
        :param seq_len: the history sequence of one data instance
        :param pred_len: the predict sequence of one data instance
        :return:
        datas: all datas ,eg:[[data sequence1],[data sequence2],[data sequence3]……]
        seqs：all sequence pairs,eg:[[50,100],[101,200]……]
        length:the length of the dataset
        """

        if os.path.exists(self.pickle_file):
            with open(self.pickle_file,'rb') as pick_file:
                data_dict = pk.load(pick_file)
                datas = data_dict['datas']
                seqs = data_dict['seqs']
                length = data_dict['length']
                #todo get other variant
        else:
            datas = []  #all datas
            seqs = []   #all seqs
            start_num = 0  # seqs start number
            for file in os.listdir(self.root_dir):
                if file.endswith('csv'):
                    full_file = os.path.join(self.root_dir, file)
                    try:
                        data = pd.read_csv(full_file, header=0, index_col=None, usecols=None)
                        del_index = set([])  # indexes of invalid data
                        #get nan index
                        nan_index = np.where(np.isnan(data))[0]
                        for i_t in nan_index:
                            del_index.add(i_t)
                        # print("nan_index = {}".format(del_index))
                        # del_index = del_index.add(nan_index)

                        for col in threshold:
                            t = set(data[data[col] < threshold[col][0]].index)
                            del_index |= t
                            t = set(data[data[col] > threshold[col][1]].index)
                            del_index |= t
                            t = set(data[data[col] == -9999].index)
                            del_index |= t



                        del_index = sorted(del_index)
                        data = data.drop(del_index)
                        total_len = len(data)  # the length of an effective sequence
                        e_seq = [[0,total_len-1]]
                        datas, seqs, start_num = self._final_data(data.values, e_seq, datas, seqs, seq_len + pred_len,
                                                             start_num)
                    except FileNotFoundError:
                        print("{} not exist".format(file))
                    else:
                        print("{} is processed".format(file))

            length = start_num
            if not os.path.exists(self.pickle_file):
                with open(self.pickle_file, 'wb') as pick_file:
                    data_dict = {'datas': datas,
                                 'seqs': seqs,
                                 'length': length}
                    pk.dump(data_dict, pick_file)
        return datas, seqs, length

    def _effective_seq(self,seq_len, pred_len, total_len, del_index):
        """
         get the effective sequence pairs for each csv file
        :param seq_len: the history sequence of one data instance
        :param pred_len: the predict sequence of one data instance
        :param total_len: total length of the data
        :param del_index: the index of data needed to deleted
        :return:
            effective_seq: the effective sequence pairs of the data sequencce eg:[[50,100],[200,300],……]
                include the start and end index
        """
        seq_pred_len = seq_len + pred_len
        effective_seq = []

        if 0 not in del_index:
            del_index.insert(0, -1)
        if (total_len - 1) not in del_index:
            del_index.insert((len(del_index)), (total_len))

        # get effective sequence
        for i in range(len(del_index)-1):
            if (del_index[i + 1] - del_index[i] - 1) >= seq_pred_len:
                effective_seq.append([del_index[i] + 1, del_index[i + 1] - 1])
        return effective_seq

    def _final_data(self,data, e_seq, datas, seqs, total_len, start_num):
        """
        :param data:the original data from the csv file
        :param e_seq:the effectiove sequence pairs from self._effective_seq() function
        :param datas: this variant contain all the effective data from the csv files
        :param seqs:this variant contains all valid independent data sequence pairs.eg[[0,10],[11,100],[101,300]……]
        :param total_len:a complete predicted sequence length containing historical and predictive sequences
        :param start_num:remember the starting number of each time
        :return:
            datas:the updated datas
            seqs:the updated seqs (include the start and end index)
            start_num:updated starting number
        """
        for temp in e_seq:
            datas.append(data[temp[0]:(temp[1]+1)])
            num = temp[1] + 1 - temp[0] - total_len + 1
            seqs.append([start_num, start_num + num - 1])
            start_num = start_num + num
        return datas, seqs, start_num

    def _save_data(self,data,path,file):
        temp = []
        for x in data:
            for y in x:
                temp.append(y)
        df = pd.DataFrame(temp)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs((path))
        print("new path = {}".format(os.path.join(path,file)))
        df.to_csv(os.path.join(path,file),sep=',')