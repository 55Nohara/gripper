import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_PATH = "C:\\Users\\imout\\Desktop\\study\\gripper\\data"
MAP_DATA = "PWMvsStiff"


def main():
    # CSVファイルの読み込み
    df = pd.read_csv(DATA_PATH + "\\" + MAP_DATA + ".csv")

    # データの取得
    x_data = df.iloc[:, 0].values 
    y_data = df.iloc[:, 1].values  

    # データのプロット
    plt.scatter(x_data, y_data, label='Data Points')
    plt.xlabel('x')
    plt.ylabel('y')

    # 4次の多項式で近似
    coefficients = np.polyfit(x_data, y_data, 4)
    poly_fit = np.poly1d(coefficients)

    # 近似曲線をプロット
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = poly_fit(x_fit)
    plt.plot(x_fit, y_fit, color='red', label='4th Degree Polynomial Fit')

    # 係数の表示
    coefficients_str = [f'{coeff:.2e}' for coeff in coefficients]
    coefficients_label = ', '.join([f'a{i}={coeff_str}' for i, coeff_str in enumerate(coefficients_str)])
    plt.text(0.5, 0.9, f'Fit Coefficients: {coefficients_label}', transform=plt.gca().transAxes, fontsize=10, color='blue')

    plt.legend()
    plt.savefig(DATA_PATH + "\\" + "PWM_STIFF_MAP.png", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
    