# -*- coding: utf-8 -*-
import nidaqmx
import numpy as np
import time
import matplotlib.pyplot as plt
from windfreak import SynthHD
import csv
import os
from datetime import datetime
from ODMR_fitting_tool import quick_find_center

class ODMRScanner:
    def __init__(self, com_port='COM6', dev_channel="Dev2/ai0"):
        self.com_port = com_port
        self.dev_channel = dev_channel
        self.synth = None
        self.task = None

    def connect(self):
        print(f"Connecting to Windfreak on {self.com_port}...")
        self.synth = SynthHD(self.com_port)
        self.synth.init()
        
        print(f"Initializing DAQ on {self.dev_channel}...")
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(self.dev_channel)
        self.task.start()

    def disconnect(self):
        if self.synth:
            self.synth[0].enable = False
            self.synth.close()
        if self.task:
            self.task.stop()
            self.task.close()
        print("Disconnected hardware.")

    def scan(self, f_start, f_end, num_points, num_repeats, power_dbm, folder_name, 
             auto_center=False, check_interval=10, shift_threshold_hz=1e6,
             file_prefix="CW"):
        # Initial range
        current_f_start = f_start
        current_f_end = f_end
        span = f_end - f_start
        
        date_str = datetime.now().strftime('%y%m%d')
        output_dir = os.path.join("./DATA/", date_str, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime('%H%M%S')
        file_name = f"{file_prefix}_{timestamp}.csv"
        file_path = os.path.join(output_dir, file_name)

        f_list = np.linspace(current_f_start, current_f_end, num_points)
        fluo_sum = np.zeros(len(f_list))
        
        # Track history of centers
        center_history = []

        self.synth[0].power = power_dbm
        self.synth[0].enable = True

        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(f_list, fluo_sum, 'b-')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Fluorescence (V)")
        plt.show(block=False)

        try:
            for n in range(num_repeats):
                # Measurement
                for i, f in enumerate(f_list):
                    self.synth[0].frequency = f
                    data = self.task.read()
                    fluo_sum[i] += data

                # Update plot
                current_avg = fluo_sum / (n + 1)
                line.set_xdata(f_list) # Frequency list might have changed
                line.set_ydata(current_avg)
                ax.set_title(f"CW Iteration: {n+1}/{num_repeats}")
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)

                # Tracking Logic
                if auto_center and (n + 1) % check_interval == 0:
                    found_center = quick_find_center(f_list, current_avg, mode='CW')
                    current_center = (current_f_start + current_f_end) / 2
                    shift = found_center - current_center
                    
                    if abs(shift) > shift_threshold_hz:
                        print(f"Tracking: Peak shifted by {shift/1e6:.2f} MHz. Adjusting window...")
                        current_f_start += shift
                        current_f_end += shift
                        f_list = np.linspace(current_f_start, current_f_end, num_points)
                        center_history.append((n+1, found_center))
                        # Note: We don't clear fluo_sum, which introduces some 'drag'.
                        # For more aggressive tracking, one could clear or weigh recent data.

            # Save data
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(list(zip(f_list, fluo_sum / num_repeats)))
            
            # Save tracking log if needed
            if center_history:
                with open(file_path.replace('.csv', '_tracking.txt'), 'w') as f:
                    for entry in center_history:
                        f.write(f"Iteration {entry[0]}: Adjusted Center to {entry[1]/1e9:.6f} GHz\n")

            print(f"Data saved to {file_path}")
            return file_path

        finally:
            self.synth[0].enable = False
            plt.ioff()
            plt.show()

    def fm_scan(self, f_start, f_end, num_points, num_repeats, power_dbm, folder_name, delta_f, 
                auto_center=False, check_interval=10, shift_threshold_hz=1e6,
                file_prefix="FM"):
        """
        Software Frequency Modulation scan with Tracking.
        """
        current_f_start = f_start
        current_f_end = f_end
        
        date_str = datetime.now().strftime('%y%m%d')
        output_dir = os.path.join("./DATA/", date_str, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime('%H%M%S')
        file_name = f"{file_prefix}_{timestamp}.csv"
        file_path = os.path.join(output_dir, file_name)

        f_list = np.linspace(current_f_start, current_f_end, num_points)
        diff_fluo_sum = np.zeros(len(f_list))
        center_history = []

        self.synth[0].power = power_dbm
        self.synth[0].enable = True

        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(f_list, diff_fluo_sum, 'r-')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Diff Fluorescence (V)")
        ax.set_title(f"FM Scan: {folder_name} (df={delta_f/1e6:.1f} MHz)")
        plt.show(block=False)

        try:
            for n in range(num_repeats):
                for i, f in enumerate(f_list):
                    self.synth[0].frequency = f + delta_f / 2
                    v_high = self.task.read()
                    self.synth[0].frequency = f - delta_f / 2
                    v_low = self.task.read()
                    diff_fluo_sum[i] += (v_high - v_low)

                # Update plot
                current_avg = diff_fluo_sum / (n + 1)
                line.set_xdata(f_list)
                line.set_ydata(current_avg)
                ax.set_title(f"FM Iteration: {n+1}/{num_repeats}")
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)

                # Tracking Logic
                if auto_center and (n + 1) % check_interval == 0:
                    found_center = quick_find_center(f_list, current_avg, mode='FM')
                    current_center = (current_f_start + current_f_end) / 2
                    shift = found_center - current_center
                    
                    if abs(shift) > shift_threshold_hz:
                        print(f"Tracking (FM): Peak shifted by {shift/1e6:.2f} MHz. Adjusting window...")
                        current_f_start += shift
                        current_f_end += shift
                        f_list = np.linspace(current_f_start, current_f_end, num_points)
                        center_history.append((n+1, found_center))

            # Save data
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(list(zip(f_list, diff_fluo_sum / num_repeats)))
            
            if center_history:
                with open(file_path.replace('.csv', '_tracking.txt'), 'w') as f:
                    for entry in center_history:
                        f.write(f"Iteration {entry[0]}: Adjusted Center to {entry[1]/1e9:.6f} GHz\n")

            print(f"FM Data saved to {file_path}")
            return file_path

        finally:
            self.synth[0].enable = False
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    # Test block
    scanner = ODMRScanner()
    try:
        # Note: This will fail if hardware is not connected, but structured for real use.
        # scanner.connect()
        # scanner.scan(2.82e9, 3.02e9, 100, 10, 5, "./DATA/test")
        pass
    finally:
        # scanner.disconnect()
        pass
