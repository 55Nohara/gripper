import os
import pandas as pd
import numpy as np
from Gather_Data import CONTROL_FREQ, CONTROL_TIME


DATA_PATH = "C:\\Users\\Matsushima\\Desktop\\study\\gripper\\data"
# DATA_PATH = "C:\\Users\\imout\\Desktop\\study\\gripper\\data"
SENSOR_CSV_NAME = "RAW_TRAIN_DATA"
D_AUGUMENTATION = "DATA_AUG"
LABEL_CSV_NAME = "LABEL"
L_AUGUMENTATION = "LABEL_AUG"

multiple = 2

def add_noise(raw_data, noised_data , i):
    noised_data[i, :, 0] = raw_data[i, :, 0] + np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))
    noised_data[i, :, 1] = raw_data[i, :, 1] + 0.01 * np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))
    noised_data[i, :, 2] = raw_data[i, :, 2] + 0.01 * np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))

    return noised_data


def main():
    try:
            os.makedirs(DATA_PATH, exist_ok=True)
    
    except OSError as e:
            print(f"Error creating directory: {e}")
    original_csv_path = os.path.join(os.getcwd(), DATA_PATH, SENSOR_CSV_NAME + ".csv")
    label_csv_path = os.path.join(os.getcwd(), DATA_PATH, LABEL_CSV_NAME + ".csv")
    new_csv_path = os.path.join(os.getcwd(), DATA_PATH, D_AUGUMENTATION + ".csv")
    new_label_path = os.path.join(os.getcwd(), DATA_PATH, L_AUGUMENTATION + ".csv")
    original_data = pd.read_csv(original_csv_path)
    label = pd.read_csv(label_csv_path)
    reshaped = original_data.values.reshape((-1, int(CONTROL_FREQ * CONTROL_TIME), 3))
    object_num = reshaped.shape[0]
    print(reshaped.shape)
    NOISED = np.zeros((object_num* multiple, int(CONTROL_FREQ* CONTROL_TIME), 3)) # (N* mul) * T * 3
    Label_Aug = np.zeros((int(CONTROL_FREQ* CONTROL_TIME), object_num* multiple))
    NOISED_reshape = np.zeros((int(CONTROL_FREQ* CONTROL_TIME), 3 *object_num* multiple))
    for i in range(multiple):
        for j in range(object_num):
            
            NOISED[object_num*i + j, :, 0] = reshaped[i, :, 0] + np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))
            NOISED[object_num*i + j, :, 1] = reshaped[i, :, 1] + 0.01 * np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))
            NOISED[object_num*i + j, :, 2] = reshaped[i, :, 2] + 0.01 * np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))
            # NOISED[2*i +1, :, 0] = reshaped[i, :, 0] + np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))
            # NOISED[2*i +1, :, 1] = reshaped[i, :, 1] + 0.01 * np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))
            # NOISED[2*i +1, :, 2] = reshaped[i, :, 2] + 0.01 * np.random.rand( int(CONTROL_FREQ* CONTROL_TIME))        
            Label_Aug[:, object_num*i + j] = label.values[:, j]
            # Label_Aug[:, 2*i +1] = label.values[:, i]


    NOISED_reshape = NOISED.reshape((int(CONTROL_FREQ * CONTROL_TIME), -1))
    df_noised = pd.DataFrame(NOISED_reshape)
    df_label_aug = pd.DataFrame(Label_Aug)
    
    updated_df = pd.concat([original_data, df_noised], axis =1)
    updated_Label_df = pd.concat([label, df_label_aug], axis =1)
    updated_df.to_csv(new_csv_path, index=False, mode='a', header=True)
    updated_Label_df.to_csv(new_label_path, index=False, mode='a', header=True)

if __name__ == '__main__':
    main()