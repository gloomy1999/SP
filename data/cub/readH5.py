import h5py
import numpy as np
import scipy.io as io
import os

# data_info = h5py.File('data_info.h5','r')
# nc = data_info['all_att'][...].shape[0]
# sf_size = data_info['all_att'][...].shape[1]
# semantic_data = {'seen_class': data_info['seen_class'][...],
#                  'unseen_class': data_info['unseen_class'][...],
#                  'all_class': np.arange(nc),
#                  'all_att': data_info['all_att'][...]}

# true_label=np.array([112,106,19,127,34,57])
# for i, class_i in enumerate(semantic_data['seen_class']):
#     idx = np.where(true_label == class_i)[0]
#     if idx or idx==0:
#         print(idx)
#         print(true_label[idx])
#         print(i)
#         print("-----------------------------------------------------")
# print(nc)
# print("-----------------------------------------------------")
# print(semantic_data['all_att'])
# print(type(semantic_data['seen_class']))
# print("-----------------------------------------------------")
# print(semantic_data['unseen_class'])
# print(type(semantic_data['unseen_class']))

def checkfile(datapath):
    assert os.path.exists(datapath), 'This is no file %s'%(datapath)
    return datapath

att_data = io.loadmat('att_splits.mat')
# print(att_data)
data_info = h5py.File('data_info.h5','r')
nc = data_info['all_att'][...].shape[0]
semantic_data = {'seen_class': data_info['seen_class'][...],
                 'unseen_class': data_info['unseen_class'][...],
                 'all_class': np.arange(nc),
                 'all_att': data_info['all_att'][...]}
print(semantic_data)
