# -*- coding: utf-8 -*-
import json
import os
import sys
from ODMR_DAC_WF_modular import ODMRScanner
from ODMR_fitting_tool import fit_odmr, plot_results, calculate_pressure

def main():
    # Load config
    config_path = 'config.json'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Hardware settings
    scanner = ODMRScanner(
        com_port=config['hardware']['com_port'],
        dev_channel=config['hardware']['dev_channel']
    )
    
    params = config['scan_params']
    track_params = config.get('tracking', {})
    mode = params.get('mode', 'CW')
    
    try:
        scanner.connect()
        if mode == 'FM':
            data_file, center_history = scanner.fm_scan(
                f_start=params['f_start'],
                f_end=params['f_end'],
                num_points=params['num_points'],
                num_repeats=params['num_repeats'],
                power_dbm=params['power_dbm'],
                folder_name=params['folder_name'],
                delta_f=params['delta_f'],
                auto_center=track_params.get('enable_auto_center', False),
                check_interval=track_params.get('check_interval', 10),
                shift_threshold_hz=track_params.get('shift_threshold_hz', 1e6),
                file_prefix=params['file_prefix']
            )
        else:
            data_file, center_history = scanner.scan(
                f_start=params['f_start'],
                f_end=params['f_end'],
                num_points=params['num_points'],
                num_repeats=params['num_repeats'],
                power_dbm=params['power_dbm'],
                folder_name=params['folder_name'],
                auto_center=track_params.get('enable_auto_center', False),
                check_interval=track_params.get('check_interval', 10),
                shift_threshold_hz=track_params.get('shift_threshold_hz', 1e6),
                file_prefix=params['file_prefix']
            )
    except Exception as e:
        print(f"Measurement error: {e}")
        sys.exit(1)
    finally:
        scanner.disconnect()

    # 2. Fitting
    num_peaks = config['fitting'].get('num_peaks', 1)
    popt, perr, x, y = fit_odmr(data_file, mode=mode, num_peaks=num_peaks)
    
    # 3. Save Results
    if popt is not None:
        result_file = data_file.replace('.csv', '_fit.txt')
        
        if num_peaks == 1:
            center = popt[0]
        elif num_peaks == 2:
            center = (popt[0] + popt[3]) / 2
        else:
            centers = popt[0:-1:3]
            center = (min(centers) + max(centers)) / 2
            
        if track_params.get('enable_auto_center', False) and center_history:
            true_center = sum([c for n, c in center_history]) / len(center_history)
        else:
            true_center = center
            
        pressure = calculate_pressure(true_center, f0_hz=config['fitting'].get('f0_reference_hz', 2.870e9))
        
        if mode == 'CW':
            offset = popt[-1]
            contrasts = [popt[i]/offset * 100 for i in range(2, len(popt)-1, 3)]
        else:
            contrasts = [popt[i] for i in range(2, len(popt)-1, 3)]
        
        with open(result_file, 'w') as f:
            f.write(f"Mode: {mode}\n")
            f.write(f"Peaks: {num_peaks}\n")
            f.write(f"Apparent Center Frequency (from fit): {center/1e9:.6f} GHz\n")
            f.write(f"True Average Center Frequency: {true_center/1e9:.6f} GHz\n")
            f.write(f"Estimated Pressure: {pressure:.4f} GPa\n")
            if mode == 'CW':
                for i, c in enumerate(contrasts):
                    f.write(f"Peak {i+1} Contrast: {c:.2f} %\n")
            else:
                for i, c in enumerate(contrasts):
                    f.write(f"Peak {i+1} FM Amplitude: {c:.4e}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Parameters: {popt.tolist()}\n")
            f.write(f"Errors: {perr.tolist()}\n")
        
        plot_file = data_file.replace('.csv', '.png')
        if config['fitting']['auto_plot']:
            plot_results(x, y, popt, mode=mode, num_peaks=num_peaks, output_path=plot_file)

    print("\nWorkflow completed successfully.")
    print(f"Raw Data: {data_file}")
    if popt is not None:
        print(f"Fit Results and Pressure saved to: {result_file}")
        print(f"Plot saved to: {plot_file}")

if __name__ == "__main__":
    main()
