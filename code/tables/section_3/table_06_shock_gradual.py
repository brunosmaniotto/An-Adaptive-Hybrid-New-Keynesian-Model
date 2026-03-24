"""
Table 6: Shock Therapy vs Gradualism
====================================

Paper Section: 3.2
Description: Generates Table 6 comparing shock therapy, gradualism, and moderate policy.
Dependencies:
    - Input: output/simulations/section_3/shock_vs_gradual.pkl
Output:
    - output/tables/section_3/table_06.csv
"""

import sys
import pickle
import pandas as pd
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def generate_table():
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'shock_vs_gradual.pkl'
    
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run simulation first.")
        return

    with open(input_path, 'rb') as f:
        results = pickle.load(f)

    data = []
    for name, res in results.items():
        rec = f"{res['recovery']}" if res['recovery'] else "Never"
        data.append({
            'Strategy': name,
            'Welfare Loss': res['welfare'],
            'Recovery Time': rec,
            'Final Theta': res['final_theta']
        })

    df = pd.DataFrame(data)
    
    output_dir = PROJECT_ROOT / 'output' / 'tables' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'table_06.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Table 6 saved to {output_path}")
    
    # Print LaTeX format for convenience
    print("\n% LaTeX table format:")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Strategy & Welfare Loss & Recovery Time & Final $\theta$ \\")
    print(r"\midrule")
    for _, row in df.iterrows():
        print(f"{row['Strategy']} & {row['Welfare Loss']:.4f} & {row['Recovery Time']} & {row['Final Theta']:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    generate_table()
