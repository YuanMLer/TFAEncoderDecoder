#endoring:utf-8

import netCDF4 as nc
import numpy as np
import pandas as pd
import os


ori_testb1_data = nc.Dataset('2018ai_ml.nc')
save_path = "now_testB"

ori_data = ori_testb1_data

ori_dimentsions,ori_variables = ori_data.dimensions,ori_data.variables

date_num = ori_dimentsions['date'].size
station_num = ori_dimentsions['station'].size
# print(data_num)
# print(station_num)
factor_list = list(ori_variables.keys())[3:]
# print(factor_list)
# print(list(ori_variables.keys()))
coder = np.eye(10)
data = []
for ind in factor_list:
    tmp = ori_variables[ind]   #获取特定要观测的数据集对应的属性信息
    tmp_data = tmp[:]
    data.append(tmp_data)


def write_data_split_by_station(data, sta, factor_list,day_points):
    csv_data = np.zeros((date_num * day_points, len(factor_list) + 9 + 2))
    for fa_data, i in zip(data, range(len(factor_list))):
        fa_sta_data = fa_data[:, :day_points, sta]
        col_data = np.array(fa_sta_data)

        csv_data[:, i] = col_data.reshape(-1)
    for i in range(csv_data.shape[0]):
        csv_data[i, -1] = i
        csv_data[i, -11:-1] = coder[sta, :]

    col_sta = list('sta%d' % i for i in range(1, 11))
    df = pd.DataFrame(csv_data, columns=factor_list + col_sta + ['time'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    filename = "./%s/testB_%d_sta%d-%d.csv" % (save_path,date_num, sta,day_points)

    print("write success for %s" % filename)
    df.to_csv(filename, index=False, sep=',')

def copy_last_13_data(f_data_24_name,f_data_37_name,factor_list):
    col_sta = list('sta%d' % i for i in range(1, 11))
    df_24 = pd.read_csv(f_data_24_name)
    df_37 = pd.read_csv(f_data_37_name)

    # df_37 = df_37.values

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    n = len(df_37)
    # print(type(df_37))
    temp = df_37[n-13:n]


    df_24 = pd.concat((df_24,temp),axis=0)
    # t_value=[1006,16,3,6,300,25,5,1,0]
    # m = len(df_24)
    # for i in range(29,38):
    #     df_24[m-33:m,i] = t_value
    file_name = "./%s/All_testB_%d_sta%d-%d_13.csv" % (save_path,date_num, sta,24)
    df_24.to_csv(file_name,index=False, sep=',')






for sta in range(station_num):
    day_points = 24
    write_data_split_by_station(data, sta, factor_list,day_points)
    day_points = 37
    write_data_split_by_station(data, sta, factor_list, day_points)
    f_data_24_name = "./%s/testB_%d_sta%d-%d.csv"% (save_path,date_num, sta,24)
    f_data_37_name = "./%s/testB_%d_sta%d-%d.csv" % (save_path,date_num, sta,37)
    print(f_data_37_name)
    copy_last_13_data(f_data_24_name, f_data_37_name,factor_list)

