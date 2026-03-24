"""
Table 5: The Credibility Buffer (Riding the Wave)
=================================================

Paper Section: 3.1
Description: Generates Table 5 comparing Passive vs Active policy under High/Low theta.
Dependencies:
    - Input: output/simulations/section_3/table_05_data.pkl
Output:
    - output/tables/section_3/table_05.csv
"""

import sys
import pickle
import pandas as pd
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def generate_table():
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'table_05_data.pkl'
    
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run simulation first.")
        return

    with open(input_path, 'rb') as f:
        results = pickle.load(f)

    data = []
    labels = ['High θ, Passive', 'High θ, Active', 'Low θ, Passive', 'Low θ, Active']
    
    for label in labels:
        res = results[label]
        
        # Calculate policy effect
        if 'Passive' in label:
            effect = "---"
        else:
            base_label = label.replace('Active', 'Passive')
            base_welfare = results[base_label]['welfare']
            pct_change = (res['welfare'] - base_welfare) / base_welfare * 100
            word = "Hurts" if pct_change > 0 else "Helps"
            effect = f"{word} ({pct_change:+.0f}%)"

        data.append({
            'Scenario': label,
            'Peak Inflation': f"{res['peak_pi']:.1f}%",
            'Welfare Loss': res['welfare'],
            'Final Theta': res['final_theta'],
            'Policy Effect': effect
        })

    df = pd.DataFrame(data)
    
    output_dir = PROJECT_ROOT / 'output' / 'tables' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'table_05.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Table 5 saved to {output_path}")
    
    # Print LaTeX format
    print("\n% LaTeX table format:")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Scenario & Peak Inflation & Welfare Loss & Final $\theta$ & Policy Effect \\")
    print(r"\midrule")
    for i, row in df.iterrows():
        print(f"{row['Scenario']} & {row['Peak Inflation']} & {row['Welfare Loss']:.4f} & {row['Final Theta']:.3f} & {row['Policy Effect']} \\\\")
        if i == 1: print(r"\\[-0.5ex]")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    generate_table()
