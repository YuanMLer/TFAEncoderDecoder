# -*- encoding utf-8 -*-
import torch
import numpy as np
import os
import sys
import torch.utils.data as data
import pandas as pd
import torch.backends.cudnn as cudnn

sys.path.append("..")
from utils import ParserList, AverageMeter
import models as models
from dataset import WeatherDatasetMoreDataTest
from losses import TotalVarMSEloss
from trainandtest import *
from Attn import Attn
import numpy as np

# 设置args
def setargs(args, strs,batchsize=1):
    args.encoder_num_layer = int(strs[2])
    args.decoder_num_layer = int(strs[4])
    args.seq_len = int(strs[8])
    args.pred_len = int(strs[10])
    args.encoder_hidden_dim = int(strs[12])
    args.decoder_hidden_dim = int(strs[12])
    args.bidirectional = False if strs[-1] == "1" else True
    args.criterion = strs[5]
    args.optimizer = strs[6]
    args.batch_size = batchsize
    # print("args.encoder_num_layer = {}".format(args.encoder_num_layer))
    # print("args.decoder_num_layer = {}".format(args.decoder_num_layer))
    # print("args.seq_len = {}".format(args.seq_len))
    # print("args.pred_len = {}".format(args.pred_len))
    # print("args.encoder_hidden_dim = {}".format(args.encoder_hidden_dim))
    # print("args.decoder_hidden_dim = {}".format(args.decoder_hidden_dim))
    # print("args.bidirectional = {}".format(args.bidirectional))
    # print("args.criterion = {}".format(args.criterion))
    # print("args.optimizer = {}".format(args.optimizer))
    return args

# 读取模型
def get_model(args, model_name):
    """
    return model
    :param args:
    :param model_name:
    :return:
    """
    model = models.__dict__[model_name](args=args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        model = model.cuda()
    return model

def eval(model, model_file, csv_file, csv_result_path, args, threshold, getAttention=False):
    """
    load model,read csv_file and write the result in the csv_result_path
    :param model:
    :param model_file:
    :param csv_file:
    :param csv_result_path:
    :return:
        num: the element number
        mse_losses:
        thetaMSEloss:
    """
    mse_loss = AverageMeter()
    total_mse_loss = AverageMeter()
    assert os.path.isfile(model_file), "{} is not a file name".format(model_file)
    print("model file is {}".format(model_file))
    checkpoint = torch.load(model_file)
    # print("checkpoint[state_dict] = {}".format(checkpoint['state_dict']))
    model.load_state_dict(checkpoint['state_dict'])

    # load dataset file
    test_dataset = WeatherDatasetMoreDataTest(csv_file, args, threshold)
    # todo how to solve the problem when we use a batch size is not 1
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    mse_criterion = torch.nn.MSELoss()
    total_mse_criterion = TotalVarMSEloss()
    # todo mae and rmse
    # mae_criterion =
    # rmse_criterion =
    # mse_test_loss, total_mse_test_loss, num, predict_M_data, obs_data, pred_out_data = \
    #     test_return_attention_data(test_loader, model, mse_criterion, total_mse_criterion, args)
    if getAttention:
        mse_test_loss, total_mse_test_loss, num, predict_M_data, obs_data, pred_out_data, attentions_data = \
            test_return_data(test_loader, model, mse_criterion, total_mse_criterion, args, getAttention)
        # write attentions
        np.save(os.path.join(csv_result_path, "atten.npy"), attentions_data)
        # draw attention images
        # attentions_data
        length = len(attentions_data)
        num = length // args.pred_len

        for i in range(num):
            data = attentions_data[i * args.pred_len:(i + 1) * args.pred_len]
            width = 800
            height = int(width * (args.pred_len / args.seq_len))
            att = Attn(height, width)
            img_name = 'weight_{}.jpg'.format(i)
            print(img_name + " is writen")
            weight_dir = os.path.join(csv_result_path, 'weight')
            if not os.path.isdir(weight_dir):
                os.makedirs(weight_dir)
            att.save(np.array(data), os.path.join(weight_dir, img_name))
    else:
        mse_test_loss, total_mse_test_loss, num, predict_M_data, obs_data, pred_out_data = \
            test_return_data(test_loader, model, mse_criterion, total_mse_criterion, args, getAttention)
    print("mse_loss = {}\n total_mse_loss = {}".format(mse_test_loss, total_mse_test_loss))
    mse_loss.update(mse_test_loss, num)
    total_mse_loss.update(total_mse_test_loss,num)
    pred_file = "pred_file.csv"
    obs_file = "obs_file.csv"
    supper_calculate_file = "supper_cal_file.csv"
    file_data_list = [(pred_file, pred_out_data),
                      (supper_calculate_file, predict_M_data),
                      (obs_file, obs_data)]
    #save csv files
    for filename, datas in file_data_list:
        datas = np.array(datas)
        # print("datas={}".format(datas))
        d = {'t2m': datas[:, 0], 'w10m': datas[:, 1], 'rh2m': datas[:, 2]}
        # print("t2m_M = {}".format(datas[:,0]))
        df = pd.DataFrame(d)
        # print(df)
        temp_file = os.path.join(csv_result_path, filename)
        df.to_csv(temp_file)
        print("*" + "save to temp_file = {}".format(temp_file) + "*")
        loss_file = os.path.join(csv_result_path, "loss.txt")
        with open(loss_file, 'w') as temp_loss_file:
            txt = "mse_test_loss = {}\ntotal_mse_test_loss = {}".format(mse_test_loss,
                                                                        total_mse_test_loss)
            txt = txt + '\n Total params:%.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0)
            temp_loss_file.write(txt)

    # save plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.array(obs_data)[-72:-1, 0],'-r',label='t2m_O')
    plt.plot(np.array(pred_out_data)[-72:-1, 0], '-b', label='t2m_P')
    plt.plot(np.array(predict_M_data)[-72:-1, 0],'-g', label='t2m_M')
    plt.legend()
    plt.savefig(os.path.join(csv_result_path,'t2m.png'))
    plt.figure()
    plt.plot(np.array(obs_data)[-72:-1, 1], '-r', label='w10m_O')
    plt.plot(np.array(pred_out_data)[-72:-1, 1], '-b', label='w10m_P')
    plt.plot(np.array(predict_M_data)[-72:-1, 1], '-g', label='w10m_M')
    plt.legend()
    plt.savefig(os.path.join(csv_result_path, 'w10m.png'))
    plt.figure()
    plt.plot(np.array(obs_data)[-72:-1, 2], '-r', label='rh2m_O')
    plt.plot(np.array(pred_out_data)[-72:-1, 2], '-b', label='rh2m_P')
    plt.plot(np.array(predict_M_data)[-72:-1, 2], '-g', label='rh2m_M')
    plt.legend()
    plt.savefig(os.path.join(csv_result_path, 'rh2m.png'))

    return num, mse_loss.avg, total_mse_loss.avg

def main():
    #test文件夹
    # testpath = "singleStationDataTestA"
    # csvfile = "testA_27_sta{}-24.csv"
    # testpath = "singleStationDataTestB"
    # csvfile = "testB_48_sta{}-24.csv"
    test_csvfiles = [("singleStationDataTestA","testA_27_sta{}-24.csv"),("singleStationDataTestB","testB_48_sta{}-24.csv")]
    station = 10
    # 读取文件列表
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
    args, state = ParserList()
    model_base = os.path.join(args.basedir, 'checkpoint')
    # files = os.listdir(model_base)
    # files.remove("twotsaendecoder_EL_2_DL_2_MSE_SGD_SEQ_48_PRED_24_HD_64_BD_1")
    # files.remove("threetsaendecoder_EL_2_DL_2_MSE_SGD_SEQ_48_PRED_24_HD_64_BD_1")
    # files.remove("fourtsaendecoder_EL_2_DL_2_MSE_SGD_SEQ_48_PRED_24_HD_64_BD_1")
    files = [
             # 'purelstm_EL_2_DL_2_MSE_Adam_SEQ_36_PRED_12_HD_64_BD_1',
             # 'purelstm_EL_2_DL_2_MSE_Adam_SEQ_48_PRED_24_HD_64_BD_1',
             # 'purelstm_EL_2_DL_2_MSE_Adam_SEQ_72_PRED_37_HD_64_BD_1',
             # 'saendecoder_EL_2_DL_2_MSE_Adam_SEQ_36_PRED_12_HD_64_BD_1',
             # 'saendecoder_EL_2_DL_2_MSE_Adam_SEQ_48_PRED_24_HD_64_BD_1',
             # 'saendecoder_EL_2_DL_2_MSE_Adam_SEQ_72_PRED_37_HD_64_BD_1',
             # 'taendecoder_EL_2_DL_2_MSE_Adam_SEQ_36_PRED_12_HD_64_BD_1',
             # 'taendecoder_EL_2_DL_2_MSE_Adam_SEQ_48_PRED_24_HD_64_BD_1',
             # 'taendecoder_EL_2_DL_2_MSE_Adam_SEQ_72_PRED_37_HD_64_BD_1',
             # 'tsaendecoder_EL_2_DL_2_MSE_Adam_SEQ_36_PRED_12_HD_64_BD_1',
             # 'tsaendecoder_EL_2_DL_2_MSE_Adam_SEQ_48_PRED_24_HD_64_BD_1',
             # 'tsaendecoder_EL_2_DL_2_MSE_Adam_SEQ_72_PRED_37_HD_64_BD_1',
             # 'tsaendecoder_EL_2_DL_2_TotalVarMSEloss_Adam_SEQ_36_PRED_12_HD_64_BD_1',
             # 'saplusendecoder_EL_2_DL_2_MSE_Adam_SEQ_36_PRED_12_HD_64_BD_1',
             # 'tsapureendecoder_EL_2_DL_2_MSE_Adam_SEQ_36_PRED_12_HD_64_BD_1',
            'taclassicendecoder_EL_2_DL_2_MSE_Adam_SEQ_36_PRED_12_HD_64_BD_1',
            'tsaclassicendecoder_EL_2_DL_2_MSE_Adam_SEQ_36_PRED_12_HD_64_BD_1'

            ]
    for file in files:
        strs = file.split("_")
        # 重新设置args参数
        args = setargs(args, strs, 1)
        modelname = strs[0]
        for testpath, csvfile in test_csvfiles:
            for i in range(0, 10):

                model_file_name = os.path.join(args.basedir, "checkpoint", file, str(i), "checkpoint.pth.tar")
                csv_file_name = os.path.join(args.basedir, "csvfiles", testpath, str(i), csvfile.format(i))
                csv_result_path = os.path.join(args.basedir, "result",testpath,file,str(i))
                if not os.path.exists(csv_result_path):
                    os.makedirs(csv_result_path)
                model = get_model(args, modelname)
                mse_losses = AverageMeter()
                total_mse_losses = AverageMeter()
                if strs[0].endswith("endecoder"):
                    num, mse_loss, total_mse_loss = eval(model, model_file_name, csv_file_name, csv_result_path,
                                                     args, threshold, getAttention=True)
                else:
                    num, mse_loss, total_mse_loss = eval(model, model_file_name, csv_file_name, csv_result_path,
                                                         args, threshold)
                mse_losses.update(mse_loss, num)
                total_mse_losses.update(total_mse_loss, num)
            total_loss_file = os.path.join(args.basedir, "result", testpath, file, "losses.txt")

            with open(total_loss_file, "w") as total_loss_file:
                temp = "ten models average mse losses is :{}".format(mse_losses.avg)
                temp += "ten models average total mse losses is :{}".format(total_mse_losses.avg)
                total_loss_file.write(temp)


if __name__ == '__main__':
    main()



