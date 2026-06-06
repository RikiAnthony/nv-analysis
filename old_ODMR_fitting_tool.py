# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def lorentzian(x, x0, gamma, amp, offset):
    """
    Standard Lorentzian function.
    x0: Center frequency
    gamma: HWHM (Half Width at Half Maximum)
    amp: Amplitude (relative to offset)
    offset: Baseline
    """
    return offset - amp * (gamma**2 / ((x - x0)**2 + gamma**2))

def fit_odmr(file_path, model='lorentzian'):
    print(f"Fitting data from {file_path} using {model}...")
    
    # Load data
    data = pd.read_csv(file_path, header=None, names=['freq', 'fluo'])
    x = data['freq'].values
    y = data['fluo'].values

    if model == 'lorentzian':
        # Initial guesses
        x0_guess = x[np.argmin(y)]
        gamma_guess = (x.max() - x.min()) / 20
        amp_guess = y.max() - y.min()
        offset_guess = y.max()
        
        p0 = [x0_guess, gamma_guess, amp_guess, offset_guess]
        
        try:
            popt, pcov = curve_fit(lorentzian, x, y, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            return popt, perr, x, y
        except Exception as e:
            print(f"Fit failed: {e}")
            return None, None, x, y
    else:
        raise ValueError(f"Model {model} not supported.")

def plot_results(x, y, popt, model='lorentzian', output_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo', label='Data', markersize=4)
    
    if popt is not None:
        if model == 'lorentzian':
            x_fine = np.linspace(x.min(), x.max(), 500)
            y_fit = lorentzian(x_fine, *popt)
            plt.plot(x_fine, y_fit, 'r-', label=f'Fit: x0={popt[0]/1e9:.4f} GHz')
            plt.title(f"ODMR Fit Result (Center: {popt[0]/1e9:.4f} GHz)")
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fluorescence (V)")
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Test block
    # fit_odmr('test_data.csv')
    pass
