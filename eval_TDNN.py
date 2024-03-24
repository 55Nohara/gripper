import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import DataLoader
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## PARAMETERS NEEDS TO BE CHANGED ########################
from FNN_model import FNN_TDNN
from TDNN_Leaning import HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM, DATA_PATH, SAVE_FOLDER_PATH, t_input

# data path of the trained weight pth file (basically the same as DATA_PATH)

WEIGHT_DATA_PATH = "C:\\Users\\imout\\Desktop\\test\\test\\data\\"
# # data path for validation dataset
DATA_PATH = "C:\\Users\\imout\\Desktop\\test\\test\\"
# # data path to save evaluation result (graphs and RMSEs)
SAVE_FOLDER_PATH = "C:\\Users\\imout\\Desktop\\test\\test\\data\\"

# selected pth file, for the last epoch, please use "total epochs-1.pth like below"
SELECTED_EPOCH = "99.pth"

###########################################################


# validation dataset name
VALID_DATA_FILE_NAME = "valid"
# validation dataset name (label)
VALID_LABEL_FILE_NAME = "label_valid"
# file name to save validation RMSEs
RMSE_FILE_NAME = "RMSE_train"
# file name of the trained weight pth file
EXPORT_WEIGHT_FILE_NAME = "best_weights"

GRAPH_LEGENDS = ["IL Activation [-]", "RF Activation [-]", "LB Activation [-]", "GM Activation [-]", "VA Activation [-]", "SB Activation [-]"]



def main():
    os.makedirs(SAVE_FOLDER_PATH, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    labels = np.array(pd.read_csv(DATA_PATH+"\\"+VALID_LABEL_FILE_NAME+".csv", encoding='utf-8').iloc[:, 1:].values)
    preds = np.zeros(labels.shape)

    #get inputData from serial communication as numpy


    inputData = torch.from_numpy(inputData.astype(np.float32))

    FNNModel = FNN_TDNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    FNNModel.load_state_dict(torch.load(WEIGHT_DATA_PATH+"\\"+EXPORT_WEIGHT_FILE_NAME+SELECTED_EPOCH))
    FNNModel.to(device)
    FNNModel.eval()
    
    #nextActivation = torch.zeros(4)    
    with torch.no_grad():
        for i in range(labels.shape[0]):
            # print(inputData[i,:].size())
            #if i != 0:
            #    inputData[i,1] = nextActivation
            ans = FNNModel(inputData[i,:].to(device))
            preds[i,:] = ans.to('cpu').detach().numpy().copy()
            nextActivation = torch.from_numpy(preds[i,:].astype(np.float32))
    
    plotGraph(preds, labels)
    print(preds.shape[1])

    print(np.sqrt(np.mean((preds-labels)**2, axis = 0)))
    rmse = np.sqrt(np.mean((preds-labels)**2, axis = 0))
    pd.DataFrame(np.concatenate([preds, labels], axis=1)).to_csv(SAVE_FOLDER_PATH + "\\" + "predictedActivations.csv")
    pd.DataFrame(rmse).to_csv(SAVE_FOLDER_PATH + "\\" + RMSE_FILE_NAME+".csv")


def plotGraph(pred, label, showGraph=True):
    #timeArr = np.arange(0, VALID_TIME-(SAMPLE_NUM/SIM_FREQ)+(1/SIM_FREQ), 1/SIM_FREQ)
    timeArr = np.arange(0, label.shape[0], 1) 
    for i in range(pred.shape[1]):
        plt.figure()
        print(timeArr.shape, label.shape, pred.shape)
        plt.plot(timeArr, label[:,i], color='b', label="Answer")
        plt.plot(timeArr, pred[:,i], color='r', label="Prediction")
        plt.xlabel("Samples [-]", fontsize=15)
        plt.ylabel(GRAPH_LEGENDS[i], fontsize=15)
        plt.ylim(-0.1,1)
        plt.grid()
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(SAVE_FOLDER_PATH + "\\" + GRAPH_LEGENDS[i]+VALID_DATA_FILE_NAME+".png")
        if showGraph:
            plt.show()


if __name__ == "__main__":
    main()