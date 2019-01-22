# encoding:utf-8

import netCDF4 as nc

ori_data = nc.Dataset("ai_challenger_wf2018_training.nc")
# print(ori_data)

ori_dimensions,ori_variables = ori_data.dimensions,ori_data.variables
# print(ori_dimensions)
# print(ori_variables)

t2m_obs = ori_variables["t2m_obs"]
# print(t2m_obs)
# print(t2m_obs.units)
# print(t2m_obs._FillValue)
# print(t2m_obs.description)
t2m_obs_data = t2m_obs[:]
# print(t2m_obs_data.shape)
# print(t2m_obs_data)
data_index,fortime_index,station_index = 1,2,3
specified_data = t2m_obs_data[data_index,fortime_index,station_index]

# print(specified_data)
print(t2m_obs_data.shape)
print(ori_variables.keys())
print(ori_variables['foretimes'][:])
print(ori_variables['station'][:])