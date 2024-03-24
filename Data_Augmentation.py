import os
import pandas as pd
import numpy as np

DATA_PATH = "C:\\Users\\imout\\Desktop\\study\\gripper\\data"
SENSOR_CSV_NAME = "TRAIN_DATA"
multiple = 10

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def load_original_data(file_path):
    df = pd.read_csv(file_path)
    return df.iloc[:, :].values

def generate_noisy_copies(original_data, num_copies):
    data_list = [original_data]
    for _ in range(num_copies):
        copy = original_data + np.random.randn(original_data.shape[0], 2)
        data_list.append(copy)
    return np.hstack(data_list)

def save_result_data(result_data, file_path):
    pd.DataFrame(result_data).to_csv(file_path, index=False, header=False)

def main():
    data_file_path = os.path.join(DATA_PATH, SENSOR_CSV_NAME + ".csv")
    create_directory(DATA_PATH)

    original_data = load_original_data(data_file_path)
    result_data = generate_noisy_copies(original_data, multiple)

    result_file_path = os.path.join(DATA_PATH, SENSOR_CSV_NAME + "_augmented.csv")
    save_result_data(result_data, result_file_path)

if __name__ == '__main__':
    main()