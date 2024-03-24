
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



class FNN_TDNN(nn.Module):

    def __init__(self, hiddenDim, t_input):
        super(FNN_TDNN, self).__init__()
        self.t_input = t_input
        self.hiddenDim = hiddenDim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.t_input* 3, hiddenDim[0])
        self.fc2 = nn.Linear(hiddenDim[0], hiddenDim[1])
        self.fc3 = nn.Linear(hiddenDim[1], hiddenDim[2])
        self.fc4 = nn.Linear(hiddenDim[2], 1)
        #self.fc5 = nn.Linear(hiddenDim[3], 1)
        # self.bn1 = nn.BatchNorm1d(hiddenDim[0])
        # self.bn2 = nn.BatchNorm1d(hiddenDim[1])
        # self.bn3 = nn.BatchNorm1d(hiddenDim[2])
        # self.bn4 = nn.BatchNorm1d(hiddenDim[3])
        # self.bn5 = nn.BatchNorm1d(hiddenDim[4])
        # self.bn6 = nn.BatchNorm1d(hiddenDim[5])
        # self.bn7 = nn.BatchNorm1d(hiddenDim[6])
        # self.bn8 = nn.BatchNorm1d(hiddenDim[7])
        # self.fc6 = nn.Linear(hiddenDim[4], hiddenDim[5])
        # self.fc7 = nn.Linear(hiddenDim[5], hiddenDim[6])
        # self.fc8 = nn.Linear(hiddenDim[6], hiddenDim[7])
        # self.fc9 = nn.Linear(hiddenDim[7], 1)

    def forward(self, x):  
        x = self.flatten(x)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.bn1(self.fc1(x)))  
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = F.relu(self.bn4(self.fc4(x)))  
        # x = F.relu(self.bn5(self.fc5(x)))
        # x = F.relu(self.bn6(self.fc6(x)))  
        # x = F.relu(self.bn7(self.fc7(x)))
        # x = F.relu(self.bn8(self.fc8(x)))  
        # x = F.relu(self.fc9(x))
        
        return x

#内包表記にしたい
class TDNN_Dataset():
    def __init__(self, csvWithPath, fileNameData, fileNameLabel, t_input):
        data_ = pd.read_csv(csvWithPath+"\\"+fileNameData+".csv", encoding='utf-8').iloc[:, :].values
        label_ = pd.read_csv(csvWithPath+"\\"+fileNameLabel+".csv", encoding='utf-8').iloc[:, :].values
        T_scale = int(data_.shape[0]) ###T
        num_objects = int(data_.shape[1]/3) ###N
        data_ = data_.reshape((-1, T_scale, 3)) ###dim of data_: N* T* 3
        
        
        # reshape data to (N*(T-t_input-1)) * t_input* 3
        fin_datasets = np.empty((num_objects * (T_scale - t_input + 1), t_input, 3)) ### (N*(T-t_input-1)) * t_input * 3
        init_datasets = np.zeros(((num_objects * (t_input-1)), t_input, 3))
        for i in range(num_objects): ### i:object number
            for j in range(T_scale -t_input+1): ### j:
                start_col = j
                end_col = j + t_input
                fin_datasets[i*(T_scale - t_input +1) + j] = data_[i, start_col:end_col, :]
            # 0 padding
            for k in range(t_input-1):
                start_col = k
                init_datasets[i, t_input-k:t_input+1, :] = data_[i, 0:k, :]
                fin_datasets[i*(T_scale- t_input + 1) + k] = init_datasets[i, k, :]
        
        #prepare label sets
        result_list = []
        for i in range(num_objects):
            row_value = label_[i, 0]
            repeated_data = np.full((( T_scale), 1), row_value)
            result_list.append(repeated_data)

        fin_labelsets = np.concatenate(result_list, axis=1).reshape((-1,1))

        #Data_Augmentation
        

        self.data = torch.tensor(fin_datasets, dtype=torch.float32)
        self.labels = torch.tensor(fin_labelsets, dtype=torch.float32)
        #print(self.data, self.labels)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #print(self.labels.size(), type(self.labels))
        return self.data[index, :], self.labels[index, :]

    def getTensor(self):
        return self.data