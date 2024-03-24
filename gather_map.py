import os
import pandas as pd
import sys
import serial
import numpy as np
import datetime


# 後は実際の実験時に適宜修正していく感じで

CONTROL_FREQ = 20  # バルブ応答速度からPWMを決定した後、決定する。　一応オシロスコープで確認も
CALIBRATION_TIME = 2  # キャリブレーションを行う時間。暫定、実験開始後に経験的に決定
INPUT_CSV_NAME = "TRAIN_INPUT"
SENSOR_CSV_NAME = "RAW_TRAIN_DATA"
LABEL_DATA_FILE_NAME = "LABEL"
CALIBRATION_CSV_NAME = "CALIBRATION"

CONTROL_TIME = 10
OBJECT_NUM = 1 #number of object
PWM_Min = 30 #探しておく
#PWM_max = 60
Label = 30 #プログラム開始前に必ず変更しておく

DATA_PATH = "C:\\Users\\imout\\Desktop\\study\\gripper\\data\\"
# # data path to save evaluation result (graphs and RMSEs)
SAVE_FOLDER_PATH = "C:\\Users\\imout\\Desktop\\study\\gripper\\data\\"

def main():
    with serial.Serial('COM3', 115200, timeout=1) as ser:
        currentInputIndex = 0
        sendDataReady = False
        try:
            os.makedirs(DATA_PATH, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory: {e}")
           
        inputData = setInputData(OBJECT_NUM) #get inputData from TDNN
        sensorData = np.zeros((int(CONTROL_FREQ*CONTROL_TIME), 3))  # 0:pressure  1,2:strain
        offset = np.zeros(3)  # 0:pressure, 1,2:strain
        sendDataReady = False
        
        while(True):
            sendDataReady = sendDataReadyCommand(ser, 'e', sendDataReady)  # Activation, 初回の一度のみ起動
            chara = strFromArduino(ser, False)  # Arduinoからの信号, 受信してないときは-1を返す
            # region select action
            if (chara=='c'):  #offset処理のために、一定数本制御と同じ処理を行う　オフセット終了信号も必要？
                print("<Calibration Started>")
                offset = calibration(ser)
                print("<Calibration Finished>")
            #repeat t and s for several times
            elif(chara=='t'):  # センサ信号(圧力、ひずみ)を受け取って2ポートバルブ用PWMDCを送る
                sensorData[currentInputIndex, :] = getSensorData(ser, offset)
                sendControlInputToArduino(ser, inputData[currentInputIndex])
                currentInputIndex += 1
            elif chara == 's':  # 本制御開始
                startTime = datetime.datetime.now()
                print("<Data pre-input is done>" + "\nControl is started at " + str(startTime))
            elif chara == 'f':  # 本制御終了とCSVエクスポート
                endTime = datetime.datetime.now()
                print("end at " + str(endTime) + "\nControl time is " + str(endTime-startTime))

                export_train(sensorData, OBJECT_NUM)  # export current train data as CSV
                export_label(Label, OBJECT_NUM)  # export current label data as CSV
                break
            elif chara == 'q':  # Timeout Error
                print("TIMEOUT in "+str(currentInputIndex))
                sys.exit(1)
            else:  # Other Errors
                print("Serial_Exception main : "+chara)

            # endregion

# region inputCalculation
#when control realtime use tdnn
#get inputData from csv file
def setInputData(object):
    #inputData = 60 * np.ones(int(CONTROL_FREQ * CONTROL_TIME)) + 40* np.random(int(CONTROL_FREQ * CONTROL_TIME))
    #original_data = 60 * np.ones(int(CONTROL_FREQ * CONTROL_TIME)) + 40* np.random(int(CONTROL_FREQ * CONTROL_TIME))
    #inputData = np.repeat(original_data, 2)
    inputData = 60 * np.ones(int(CONTROL_FREQ * CONTROL_TIME))
    csv_path = os.path.join(os.getcwd(), DATA_PATH, f"{INPUT_CSV_NAME}_.csv")
    try:
        existing_df = pd.read_csv(csv_path)
        df_new = pd.DataFrame({f"case_{object}": inputData})
        updated_df = pd.concat([existing_df, df_new], axis =1)
        os.remove(csv_path)
        updated_df.to_csv(csv_path, index=False, mode='a', header=True)  
    except FileNotFoundError:
        pd.DataFrame(inputData, columns= [f"case_1"]).to_csv(csv_path, index=False)

    return inputData
# endregion

# region sensing
def getSensorData(ser, offset):
    sensorData = np.zeros(3)
    #sensorData[0] = calculateAngle(listener, inputIndex)
    sensorData[0], sensorData[1], sensorData[2] = getSensorDataFromArduino(ser)
    sensorData -= offset

    return sensorData

# よく考えてみたが、(簡単な)この形式で問題はないと思う。もし何かバグとか、問題が起きたら圧力データ受け取るときみたいに入力待ち信号を出す感じで。
def getSensorDataFromArduino(ser):
    pressure_data = 0.0
    strain_data_1 = 0.0
    strain_data_2 = 0.0
    tmpChar = ""

    while True:
        tmp = strFromArduino(ser, False)
        if tmp != "-1":
            tmpChar += tmp
        if tmp == '\n':
            parts = tmpChar.strip().split(',')
            try:
                pressure_data = float(parts[0])
                strain_data_1 = float(parts[1])
                strain_data_2 = float(parts[2])
            except ValueError:
                print("Error converting data to float.")

            break

    # while ser.in_waiting > 0:
        # expData = str(ser.read().decode('utf-8'))
        # print("Received Exceptional Data in get sensorData: " + expData)

    return pressure_data, strain_data_1, strain_data_2

# endregion


# region calibration
def calibration(ser):
    calibrationData = np.zeros((int(CONTROL_FREQ*CALIBRATION_TIME), 3))  # 0:pressure  1,2:strain
    press_calibration = np.zeros(1)
    strain_calibration_1 = np.zeros(1) #mean
    strain_calibration_2 = np.zeros(1) #mean
    calibrationIndex = 0

    while(True):
        chara = strFromArduino(ser, False)
        
        if (chara=='i'):
            calibrationData[calibrationIndex] = getSensorDataFromArduino(ser)
            calibrationIndex += 1
        elif (chara=='m'):
            press_calibration, strain_calibration_1, strain_calibration_2 = calculateOffsetAndExportCSV(calibrationData)
            ser.write(bytes('m', 'utf-8'))  # 文字を送ってflush()する
            ser.flush()
            break
        else:
            print("Serial_Exception in Calibration: "+chara)

    return press_calibration, strain_calibration_1, strain_calibration_2

def calculateOffsetAndExportCSV(sensorData):
    result = np.mean(sensorData, axis=0) 
    csvOffset = np.array([["pressure_offset", "strain_1_offset", "strain_2_offset"], result])
    csv_path = os.path.join(os.getcwd(), DATA_PATH, CALIBRATION_CSV_NAME + ".csv")
    pd.DataFrame(np.vstack([sensorData, csvOffset])).to_csv(csv_path)
    print("<Calibration CSV is exported>")
    return result


def export_train(sensorData, object):
    csv_path = os.path.join(os.getcwd(), DATA_PATH, SENSOR_CSV_NAME + ".csv")
    try:
        existing_df = pd.read_csv(csv_path)
        df_new = pd.DataFrame(sensorData, columns= [f"pr_{object}", f"st1_{object}", f"st2_{object}"])
        updated_df = pd.concat([existing_df, df_new], axis =1)
        os.remove(csv_path)
        updated_df.to_csv(csv_path, index=False, mode='a', header=True)  
    except FileNotFoundError:
        pd.DataFrame(sensorData, columns= [f"pr_1", f"st1_1", f"st2_1"]).to_csv(csv_path, index=False)
    print("<CSV is exported>")
# endregion

#add label data to csv file
def export_label(label, object):
    labelData = label * np.ones((int(CONTROL_FREQ * CONTROL_TIME), 1))
    csv_path = os.path.join(os.getcwd(), DATA_PATH, LABEL_DATA_FILE_NAME + ".csv")
    try:
        existing_df = pd.read_csv(csv_path)
        df_new = pd.DataFrame(labelData, columns= [f"label_{object}"])
        updated_df = pd.concat([existing_df, df_new], axis =1)
        os.remove(csv_path)
        updated_df.to_csv(csv_path, index=False, mode='a', header=True)  
    except FileNotFoundError:
        pd.DataFrame(labelData, columns= [f"label_1"]).to_csv(csv_path, index=False)
    print("<CSV is exported>")


# region function for SerialCom
def strFromArduino(serialInst, sendData=True, sendChar='w'):
    if sendData:
        # to activate serial.available()
        serialInst.write(bytes(sendChar, 'utf-8'))
        serialInst.flush()
    data = str(serialInst.read().decode('utf-8'))
    if data == '':  
        data = "-1"
    return data

def sendDataReadyCommand(serialInst, commandChar, sendDataReady):
    while(not sendDataReady):
        data = strFromArduino(serialInst, True, commandChar)
        if(data == commandChar):
            print("<SerialCom is Activated>")
            break
    return True

def sendControlInputToArduino(serialInst, target):
    serialInst.write(
        bytes(str(float(target))+'\n', 'utf-8'))
    serialInst.flush()
# endregion


if __name__ == '__main__':
    main()
