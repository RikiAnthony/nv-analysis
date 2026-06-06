# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# --- Models ---

def lorentzian(x, x0, gamma, amp, offset):
    return offset - amp * (gamma**2 / ((x - x0)**2 + gamma**2))

def double_lorentzian(x, x1, g1, a1, x2, g2, a2, offset):
    return offset - a1 * (g1**2 / ((x - x1)**2 + g1**2)) - a2 * (g2**2 / ((x - x2)**2 + g2**2))

def deriv_lorentzian(x, x0, gamma, amp, offset_deriv):
    """
    Derivative of a Lorentzian. 
    L(x) = offset - amp * (gamma**2 / ((x - x0)**2 + gamma**2))
    L'(x) = 2 * amp * gamma**2 * (x - x0) / ((x - x0)**2 + gamma**2)**2
    """
    return offset_deriv + 2 * amp * gamma**2 * (x - x0) / ((x - x0)**2 + gamma**2)**2

def double_deriv_lorentzian(x, x1, g1, a1, x2, g2, a2, offset_deriv):
    return (offset_deriv + 
            2 * a1 * g1**2 * (x - x1) / ((x - x1)**2 + g1**2)**2 + 
            2 * a2 * g2**2 * (x - x2) / ((x - x2)**2 + g2**2)**2)

# --- Analysis Functions ---

def calculate_pressure(f_center_hz, f0_hz=2.870e9):
    """
    Calculate pressure in GPa from the shift in D constant.
    dD/dP = 14.59 MHz/GPa (Doherty et al., 2014)
    """
    shift_hz = f_center_hz - f0_hz
    pressure_gpa = shift_hz / 14.59e6
    return pressure_gpa

def fit_odmr(file_path, mode='CW', num_peaks=1):
    print(f"Fitting {mode} data from {file_path} ({num_peaks} peaks)...")
    
    # Load data
    data = pd.read_csv(file_path, header=None, names=['freq', 'fluo'])
    x = data['freq'].values
    y = data['fluo'].values

    # Initial Guesses
    offset_guess = y.max() if mode == 'CW' else np.mean(y)
    amp_guess = y.max() - y.min()
    gamma_guess = (x.max() - x.min()) / 20

    try:
        if mode == 'CW':
            if num_peaks == 1:
                x0_guess = x[np.argmin(y)]
                p0 = [x0_guess, gamma_guess, amp_guess, offset_guess]
                popt, pcov = curve_fit(lorentzian, x, y, p0=p0)
            else:
                # 2 peaks
                x1_guess = x[np.argmin(y[:len(y)//2])]
                x2_guess = x[len(x)//2 + np.argmin(y[len(y)//2:])]
                p0 = [x1_guess, gamma_guess, amp_guess/2, x2_guess, gamma_guess, amp_guess/2, offset_guess]
                popt, pcov = curve_fit(double_lorentzian, x, y, p0=p0)
        else:
            # FM mode (derivative)
            if num_peaks == 1:
                x0_guess = x[np.argmax(y)] # Derivative crosses zero at x0
                # Roughly center is between peak and dip
                p0 = [x[len(x)//2], gamma_guess, amp_guess, 0]
                popt, pcov = curve_fit(deriv_lorentzian, x, y, p0=p0)
            else:
                p0 = [x[len(x)//3], gamma_guess, amp_guess/2, x[2*len(x)//3], gamma_guess, amp_guess/2, 0]
                popt, pcov = curve_fit(double_deriv_lorentzian, x, y, p0=p0)
        
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, x, y
    except Exception as e:
        # print(f"Fit failed: {e}")
        return None, None, x, y

def quick_find_center(x, y, mode='CW', num_peaks=1):
    """
    Faster and more robust center finding for real-time tracking.
    """
    if mode == 'CW':
        if num_peaks == 1:
            return x[np.argmin(y)]
        else:
            # Simple heuristic for 2 peaks
            idx_mid = len(y) // 2
            p1 = x[np.argmin(y[:idx_mid])]
            p2 = x[idx_mid + np.argmin(y[idx_mid:])]
            return (p1 + p2) / 2
    else:
        # FM: center is zero crossing
        # Rough estimate: index of max or min derivative? 
        # Actually it's between the positive and negative peaks.
        idx_max = np.argmax(y)
        idx_min = np.argmin(y)
        return (x[idx_max] + x[idx_min]) / 2

def plot_results(x, y, popt, mode='CW', num_peaks=1, output_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ko', label='Data', markersize=3, alpha=0.5)
    
    if popt is not None:
        x_fine = np.linspace(x.min(), x.max(), 1000)
        if mode == 'CW':
            if num_peaks == 1:
                y_fit = lorentzian(x_fine, *popt)
                center = popt[0]
            else:
                y_fit = double_lorentzian(x_fine, *popt)
                center = (popt[0] + popt[3]) / 2
            plt.plot(x_fine, y_fit, 'r-', lw=2, label='Fit')
        else:
            if num_peaks == 1:
                y_fit = deriv_lorentzian(x_fine, *popt)
                center = popt[0]
            else:
                y_fit = double_deriv_lorentzian(x_fine, *popt)
                center = (popt[0] + popt[3]) / 2
            plt.plot(x_fine, y_fit, 'g-', lw=2, label='FM Fit')
        
        pressure = calculate_pressure(center)
        plt.title(f"ODMR Fit ({mode}, {num_peaks} peak(s))\nCenter: {center/1e9:.4f} GHz, Est. Pressure: {pressure:.2f} GPa")
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Signal (V)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    pass
