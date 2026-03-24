"""
Appendix D2: Window Size Sensitivity
====================================

Paper Section: Appendix D
Description: Robustness check for rolling AR(1) persistence across window sizes.
Dependencies:
    - Input: data/raw/fred_data_raw.csv
Output:
    - output/tables/appendix_D/table_D2.csv
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def analyze_window_sensitivity():
    print("Appendix D2: Window Size Sensitivity Analysis...")
    
    # Load data
    fred_path = PROJECT_ROOT / 'data' / 'raw' / 'fred_data_raw.csv'
    if not fred_path.exists():
        print(f"Error: {fred_path} not found.")
        return

    df = pd.read_csv(fred_path)
    inflation = df['inflation_cpi'].dropna().values

    window_sizes = [32, 40, 48]
    data = []

    for window in window_sizes:
        rho_estimates = []
        for start in range(len(inflation) - window):
            end = start + window
            window_data = inflation[start:end]
            if np.std(window_data[:-1]) > 1e-10:
                X = sm.add_constant(window_data[:-1])
                rho = sm.OLS(window_data[1:], X).fit().params[1]
                if not np.isnan(rho):
                    rho_estimates.append(rho)
        
        rho_array = np.array(rho_estimates)
        data.append({
            'Window Size': window,
            'Mean Rho': np.mean(rho_array),
            'Std Rho': np.std(rho_array),
            'Min Rho': np.min(rho_array),
            'Max Rho': np.max(rho_array),
            'N': len(rho_estimates)
        })

    df_table = pd.DataFrame(data)
    
    output_path = PROJECT_ROOT / 'output' / 'tables' / 'appendix_D' / 'table_D2.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(output_path, index=False)
    
    print(df_table.to_string(index=False))
    print(f"\nTable D2 saved to {output_path}")

if __name__ == "__main__":
    analyze_window_sensitivity()
