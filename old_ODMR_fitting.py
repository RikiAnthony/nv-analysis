-*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:40:35 2024

@author: Arai_
"""

import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import os
import pandas as pd

Directry = "./DATA/251005/"
param_path = Directry + 'processed_fit1'
_file_name = "20251005_203010"
_save_name = _file_name
# _save_name = "27"

def lor_1(x, x0, a, hwhm, off):
    return off - a * hwhm**2 / (hwhm**2 + (x - x0)**2)

def lor_2(x, x0, a, hwhm, x02, a2, hwhm2, off):
    return off - a * hwhm**2 / (hwhm**2 + (x - x0)**2) - a2 * hwhm2**2 / (hwhm2**2 + (x - x02)**2)



def fit_and_plot(func, data):
    # fitting
    # params0 = [xdata[np.argmin(ydata)], np.ptp(ydata), 5e6, max(ydata)]
    with open(Directry + data + ".csv", "r", encoding="utf-8") as file:
      reader = csv.reader(file)
      next(reader) # Skip header row
      rows = [[float(r) for r in row] for row in reader]
      xdata = [r[0] for r in rows if r[0] != ""]
      # y1 = [r[1] for r in rows if r[1] != ""]
      # y2 = [r[2] for r in rows if r[2] != ""]
      # ydata = np.array(y2)/np.array(y1)
      ydata = [float(r[1]) for r in rows if r[0] != ""]

    params0 = [2.87e9, np.ptp(ydata), 5e6, max(ydata)]
    popt, pcov = curve_fit(func, xdata, ydata, p0=params0)
    print(np.sqrt(np.diag(pcov)))
    # スムーズなプロット用 x 値（1000点）
    x_fit = np.linspace(min(xdata), max(xdata), 1000)
    y_fit = func(x_fit, *popt)  # フィッティング曲線の値
    # パラメータの標準誤差
    perr = np.sqrt(np.diag(pcov))

    # 95% 信頼区間（標準誤差 × 1.96）
    conf_95 = 1.96 * perr

    # (x0 + x02) / 2 の標準誤差（誤差伝搬の法則）
    sigma_x0 = perr[0]


    # すべてのフィッティングパラメータ
    params_names = ["x0", "a", "hwhm", "x0", "a", "hwhm", "off", "x0"]
    params_values =  [popt[0]] + [popt[1]] + [popt[2]] + [popt[0]] + [popt[1]] + [popt[2]] + [popt[3]] + [popt[0]]# (x0 + x02) / 2
    params_errors =  [perr[0]] + [perr[1]] + [perr[2]] + [perr[0]] + [perr[1]] + [perr[2]] + [perr[3]] + [perr[0]]
    conf_95_list =  [conf_95[0]] + [conf_95[1]] + [conf_95[2]] + [conf_95[0]] + [conf_95[1]] + [conf_95[2]] + [conf_95[3]] + [conf_95[0]]

    # # フィッティング結果をCSVに保存
    output_dir = os.path.join(Directry, "processed_fit1")
    os.makedirs(output_dir, exist_ok=True)  # 保存先フォルダを作成
    output_path_params = os.path.join(output_dir, f"{_save_name}_params.csv")

    with open(output_path_params, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "Standard Error", "95% Confidence Interval"])  # ヘッダー
        writer.writerows(zip(params_names, params_values, params_errors, conf_95_list))  #


    # analysis
    f0 = (popt[0]) / 1e9
    I0 = popt[-1]
    I1 = I0 - popt[1]
    Cont = (I0 - I1) / I0
    left = 2.9e9 # Hz
    right = 3.1e9 # Hz
    I0_res = [(yi - I0)**2 for xi, yi in zip(xdata, ydata) if not left < xi < right]
    I0_var = sum(I0_res) / len(I0_res)
    I0_stddv = np.sqrt(I0_var)
    FWHM = popt[2] * 2 * 1e-9
    SNR = I0 * Cont / I0_stddv * 2
    normalized_y = ydata / I0
    normalized_y_fit = y_fit / I0


    # output
    print(f"f0: {f0:.4f} GHz")
    print(f"f0_err: {1.96 * sigma_x0  / 1e6:.6f} MHz")
    print(f"I0: {I0:.8f} ")
    print(f"I0_stddv: {I0_stddv:.5f} V")
    print(f"Cont: {Cont:.3f}")
    print(f"FWHM: {FWHM:.2f} GHz")
    print(f"SNR: {SNR:.2f}")

    print((f0-2.87)/14.5e-3)
    plt.plot(xdata, ydata,'.')
    plt.plot(x_fit, y_fit)
    plt.text(min(xdata), min(ydata), _save_name + "\n" + f"f0: {f0:.3f} GHz\n" + f"f0_err: {1.96 * sigma_x0  / 1e6:.6f} MHz\n" + f"I0_stddv: {I0_stddv:.5f} V\n" +  f"Cont: {Cont:.3f}\n" + f"FWHM: {FWHM:.2f} GHz\n" + f"SNR: {SNR:.2f}")
    plt.xlabel("freq (Hz)")
    plt.ylabel("fluo")
    plt.savefig(Directry + "processed_fit1/" + _save_name, bbox_inches="tight", dpi=800)
    plt.show()
    plt.plot(xdata, ydata / I0,'.')
    plt.plot(x_fit, normalized_y_fit)
    plt.show()
    # フィッティング結果をCSVに保存
    output_dir = os.path.join(Directry, "processed_fit1")
    os.makedirs(output_dir, exist_ok=True)  # 保存先フォルダを作成
    output_path = os.path.join(output_dir, f"{_save_name}_data.csv")

    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        # writer.writerow(["Frequency (Hz)", "Fitted Fluorescence (V)"])  # ヘッダー
        writer.writerows(zip(xdata, ydata, normalized_y))  # データを書き込み

    output_dir = os.path.join(Directry, "processed_fit1")
    os.makedirs(output_dir, exist_ok=True)  # 保存先フォルダを作成
    output_path_fit = os.path.join(output_dir, f"{_save_name}_fit1.csv")
    with open(output_path_fit, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        # writer.writerow(["Frequency (Hz)", "Fitted Fluorescence (V)"])  # ヘッダー
        writer.writerows(zip(x_fit, y_fit, normalized_y_fit))  # データを書き込み

    return 0

fit_and_plot(lor_1, _file_name)
#fit_and_plot2(lor_2, _file_name)