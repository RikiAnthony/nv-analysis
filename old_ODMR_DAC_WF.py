# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:52:39 2024

@author: Arai_
"""

from scipy.optimize import curve_fit

import nidaqmx
import numpy as np
import time
import matplotlib.pyplot as plt
from windfreak import SynthHD
import csv
import time
import os

# ==== 保存先フォルダ設定 ====
folder = "251024/test"   # ← ここを毎回変更
OUTPUT_DIR = "./DATA/" + folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Windfreak
synth = SynthHD('COM6')
synth.init()

# Variables
synth[0].power = 5  # dBm
f_start = 2.82e9  # Hz
aaa = 2e8
f_end = f_start + aaa  # Hz
num = 100  # points
N = 100  # repeats 100
_file_name = time.strftime('%Y%m%d_%H%M%S')  # csv file output
print(_file_name)

# x and y
f_list = np.linspace(f_start, f_end, num)
fluo_list = np.zeros(len(f_list))

# DAQ
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev2/ai0")
task.start()

synth[0].enable = True

# ==== プロット準備（ここだけ変更）====
plt.ion()  # インタラクティブ描画
fig, ax = plt.subplots()
(line,) = ax.plot(f_list, fluo_list, 'b-')
ax.set_xlabel("freq (Hz)")
ax.set_ylabel("fluo (V)")
ax.set_title("N = 0")
plt.show(block=False)  # ウィンドウを開いたまま進める

# for loop
for n in range(N):
    for i, f in enumerate(f_list):
        # Windfreak
        synth[0].frequency = f
        # DAQ
        data = task.read()
        fluo_list[i] += data

    # ==== ここから更新処理（変更点）====
    line.set_data(f_list, fluo_list / (n + 1))
    ax.set_title(f"N = 0 ~ {n}")
    ax.relim()             # データ範囲再計算
    ax.autoscale_view()    # 軸の再スケーリング
    plt.pause(0.01)        # イベント処理＋再描画

# Windfreak
synth[0].enable = False
synth.close()
# DAQ
task.stop()
task.close()

# Output
with open(os.path.join(OUTPUT_DIR, _file_name + ".csv"), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(list(zip(f_list, fluo_list / N)))

plt.ioff()
plt.show()
