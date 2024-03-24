import os
import sys
import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.optim


#from pytorchtools import EarlyStopping

from TDNN_model import FNN_TDNN, TDNN_Dataset



DATA_PATH = "C:\\Users\\imout\\Desktop\\study\\gripper\\data\\"
SAVE_FOLDER_PATH = "./data"

TRAIN_DATA_FILE_NAME = "RAW_TRAIN_DATA" 
TRAIN_LABEL_FILE_NAME = "LABEL"

EXPORT_WEIGHT_FILE_NAME = "best_weights"  

#L2_NORM = False
#NORM_ALPHA = 0.001

t_input = 20
BATCH_SIZE = 512 # 128 original: 100    mod 400
#try one dimension for now
HIDDEN_DIM = [100, 100, 100, 100]


EPOCH = 100
LEARNING_RATE = 0.0001  


def main():
    os.makedirs(SAVE_FOLDER_PATH, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datasets = DataLoader(TDNN_Dataset(DATA_PATH, TRAIN_DATA_FILE_NAME, TRAIN_LABEL_FILE_NAME, t_input), batch_size=BATCH_SIZE, shuffle=True)
    data_size = len(datasets)
    
    train_size = int(0.8* data_size)
    test_size = data_size - train_size
    train_dataset, test_dataset = random_split(datasets, [train_size, test_size])

    # train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #train = DataLoader(TDNN_Dataset(DATA_PATH, TRAIN_DATA_FILE_NAME, TRAIN_LABEL_FILE_NAME, t_input), batch_size=BATCH_SIZE, shuffle=True)
    #valid = DataLoader(TDNN_Dataset(DATA_PATH, VALID_DATA_FILE_NAME, VALID_LABEL_FILE_NAME, t_input), batch_size=BATCH_SIZE, shuffle=True)

    # print(train)
    # return
    startTime = datetime.datetime.now()

    TDNNmodel = FNN_TDNN(HIDDEN_DIM, t_input).to(device)
    optimizer = torch.optim.Adam(TDNNmodel.parameters(), lr=LEARNING_RATE)
    #optimizer = torch.optim.RMSprop(TDNNmodel.parameters(), lr=LEARNING_RATE, momentum=0.95)  # mod version
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
    criterion = nn.MSELoss()  # criterion = nn.L1Loss()  # in case noisy
    history = {"train_loss":[], "valid_loss":[], 'valid_RMSE':[]}
    epochLoss_val = 0
    epochLoss_train = 0


    #earlystopping = EarlyStopping(patience=500, verbose=True)

    for e in range(EPOCH):


        try:
            TDNNmodel.train()
            train_loss = 0.0
            for data, target in train_dataset.dataset:
                #if e==0:
                #    print(data.to('cpu').detach().numpy().copy().shape, target.to('cpu').detach().numpy().copy().shape)
                #    print("comeT")
                optimizer.zero_grad()
                output = TDNNmodel(data.to(device))
                loss = criterion(output, target.to(device))
                #if L2_NORM:
                #    l2 = torch.tensor(0., requires_grad=True)
                #    for w in TDNNmodel.parameters():
                #        l2 = l2+torch.norm(w)**2
                #    loss = loss + NORM_ALPHA*l2
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                #print(outPut.size(), target.size())
            epochLoss_train = train_loss/train_size

            #Evaluate at every 20 epoch
            if e % 20 == 0:
                TDNNmodel.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data_val, target_val in test_dataset.dataset:
                        #if e==0:
                        #    print(data_val.to('cpu').detach().numpy().copy().shape, target_val.to('cpu').detach().numpy().copy().shape)
                            #print("comeV")
                        outPut_val = TDNNmodel(data_val.to(device))
                        val_loss += criterion(outPut_val, target_val.to(device)).item()
                    
                epochLoss_val = val_loss/test_size


            history['train_loss'].append(epochLoss_train)
            history['valid_loss'].append(epochLoss_val)
            history['valid_RMSE'].append(math.sqrt(epochLoss_val))

            print("Epoch: " + str(e+1) + "/" + str(EPOCH) + " is finished, trainLoss: " + str(epochLoss_train) + ", valLoss: " + str(epochLoss_val), ", RMSE train:" + str(math.sqrt(epochLoss_train)), ", RMSE val:" + str(math.sqrt(epochLoss_val)))
            
            
            #scheduler.step()
            #earlystopping(epochLoss_val, TDNNmodel)
            #if earlystopping.early_stop:
            #    print("EARLY STOP")
            #    break

        except KeyboardInterrupt:

            os.makedirs(SAVE_FOLDER_PATH+"\\tmpSave\\", exist_ok=True)
            torch.save({'epoch': e,'model_state_dict': TDNNmodel.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': history,}, SAVE_FOLDER_PATH+"\\tmpSave\\tmpSave"+str(e)+"epoch.pth")
            print("saved")
            sys.exit(0)

        if (e%50 == 0 and e != 0) or e==EPOCH-1:
            torch.save(TDNNmodel.state_dict(), SAVE_FOLDER_PATH + "\\" + EXPORT_WEIGHT_FILE_NAME + str(e) +".pth")

    calcTime = datetime.datetime.now() - startTime
    exportCSV(history['train_loss'], history['valid_loss'], history['valid_RMSE'], calcTime)
    torch.save(TDNNmodel.state_dict(), SAVE_FOLDER_PATH + "\\" + EXPORT_WEIGHT_FILE_NAME + ".pth")
    plotLearningCurve(history['train_loss'], history['valid_loss'], showGraph=True)
    print("---------------------------------ML is finished---------------------------------")
    




def exportCSV(trainLoss, valLoss, valRMSE, calcTime):
    pathName = SAVE_FOLDER_PATH + "\\"
    pd.DataFrame(trainLoss).to_csv(pathName + "_train_Loss.csv")
    pd.DataFrame(valLoss).to_csv(pathName + "_validation_Loss.csv")
    pd.DataFrame(valRMSE).to_csv(pathName + "_validation_RMSE.csv")
    pd.DataFrame(np.array([calcTime])).to_csv(pathName + "_calculation_time.csv")

def plotLearningCurve(trainLoss, valLoss, showGraph=True):
    plt.figure()
    plt.plot(np.arange(EPOCH), trainLoss, color = 'r', label = "Loss: Train")
    plt.plot(np.arange(EPOCH), valLoss, color = 'b', label = "Loss: Validation")
    plt.xlabel("Epoch [-]", fontsize=15)
    plt.ylabel("Loss (MSE Loss) [-]", fontsize=15)
    #plt.ylim(0, 1)
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(SAVE_FOLDER_PATH + "\\" + "_train_Curve3d.png")
    if showGraph:
        plt.show()


if __name__ == "__main__":
    main()