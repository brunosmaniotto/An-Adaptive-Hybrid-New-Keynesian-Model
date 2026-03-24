"""
Master Replication Script
=========================

Orchestrates the full replication pipeline for "An Adaptive-Hybrid New-Keynesian Model".

Usage:
    python run_all.py              # Run full pipeline (sims -> empirical -> figures -> tables)
    python run_all.py --figures    # Only regenerate figures (assumes output/simulations exists)

Pipeline Stages:
1. Simulations: Run models, save results to output/simulations/
2. Empirical: Analyze data, save results to output/empirical/
3. Figures: Load results, generate plots in output/figures/
4. Tables: Load results, generate LaTeX/CSV tables in output/tables/
"""

import sys
import argparse
import subprocess
from pathlib import Path
import time

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = PROJECT_ROOT / 'code'

def run_script(script_path, description):
    """Execute a python script and track time."""
    print(f"\n[Running] {description}...")
    try:
        # Relative path for display
        rel_path = script_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_path = script_path

    print(f"Script: {rel_path}")

    start = time.time()
    try:
        # Run script as subprocess
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False  # Stream output to console
        )
        duration = time.time() - start
        print(f"[Done] took {duration:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Script failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Replication Master Script")
    parser.add_argument('--figures', action='store_true', help="Only run figure generation")
    parser.add_argument('--tables', action='store_true', help="Only run table generation")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. SIMULATIONS
    # ---------------------------------------------------------
    if not (args.figures or args.tables):
        print("="*60)
        print("PHASE 1: SIMULATIONS")
        print("="*60)

        sim_dir = CODE_DIR / 'simulations'

        # Section 2: The Model
        run_script(sim_dir / 'section_2/learning_mechanism.py', "Fig 1: Learning Mechanism")
        run_script(sim_dir / 'section_2/k_epsilon_roles.py', "Fig 2: Roles of k and epsilon")
        run_script(sim_dir / 'section_2/lambda_comparison.py', "Fig 3: Lambda Comparison")
        run_script(sim_dir / 'section_2/shock_regimes.py', "Fig 4: Shock Regimes")

        # Section 3: Applications
        run_script(sim_dir / 'section_3/oil_shocks.py', "Fig 6: Oil Shocks")
        run_script(sim_dir / 'section_3/great_moderation.py', "Fig 7: Great Moderation")
        run_script(sim_dir / 'section_3/missing_disinflation.py', "Fig 8: Missing Disinflation")
        run_script(sim_dir / 'section_3/post_pandemic.py', "Fig 9: Post-Pandemic")
        run_script(sim_dir / 'section_3/transmission_lags.py', "Fig 10: Transmission Lags")
        run_script(sim_dir / 'section_3/phillips_curve.py', "Fig 11: Phillips Curve")
        run_script(sim_dir / 'section_3/policy_asymmetry.py', "Fig 12: Policy Asymmetry")
        run_script(sim_dir / 'section_3/credibility_buffer.py', "Fig 13: Credibility Buffer")
        
        # Section 3: Additional Tables
        run_script(sim_dir / 'section_3/credibility_buffer_table.py', "Table 5 Simulation")
        run_script(sim_dir / 'section_3/shock_vs_gradual.py', "Table 6 Simulation")
        
        # Section 4: Extensions
        run_script(sim_dir / 'section_4/hyperinflation.py', "Fig 14: Hyperinflation")
        run_script(sim_dir / 'section_4/brazil.py', "Fig 15: Brazil")
        run_script(sim_dir / 'section_4/long_memory.py', "Fig 16: Long Memory")
        run_script(sim_dir / 'section_4/japan.py', "Fig 17: Japan")
        
        # Appendices
        run_script(sim_dir / 'appendix_B/reanchoring.py', "Fig B1: Reanchoring")
        run_script(sim_dir / 'appendix_E/mab_vs_bayesian.py', "Fig E1: MAB vs Bayesian")

    # ---------------------------------------------------------
    # 2. EMPIRICAL
    # ---------------------------------------------------------
    if not (args.figures or args.tables):
        print("\n" + "="*60)
        print("PHASE 2: EMPIRICAL ANALYSIS")
        print("="*60)

        emp_dir = CODE_DIR / 'empirical'
        run_script(emp_dir / 'section_2/persistence_analysis.py', "Fig 5: Persistence Analysis")
        
        # Appendix D Robustness
        run_script(emp_dir / 'appendix_D/kalman_comparison.py', "Table D1: Kalman Robustness")
        run_script(emp_dir / 'appendix_D/window_sensitivity.py', "Table D2: Window Sensitivity")

    # ---------------------------------------------------------
    # 3. PLOTTING
    # ---------------------------------------------------------
    if not args.tables:
        print("\n" + "="*60)
        print("PHASE 3: FIGURE GENERATION")
        print("="*60)

        plot_dir = CODE_DIR / 'plotting'

        # Section 2
        run_script(plot_dir / 'section_2/figure_01_learning_mechanism.py', "Plot Fig 1")
        run_script(plot_dir / 'section_2/figure_02_k_epsilon_roles.py', "Plot Fig 2")
        run_script(plot_dir / 'section_2/figure_03_lambda_comparison.py', "Plot Fig 3")
        run_script(plot_dir / 'section_2/figure_04_shock_regimes.py', "Plot Fig 4")
        run_script(plot_dir / 'section_2/figure_05_persistence.py', "Plot Fig 5")
        
        # Section 3
        run_script(plot_dir / 'section_3/figure_06_oil_shocks.py', "Plot Fig 6")
        run_script(plot_dir / 'section_3/figure_07_great_moderation.py', "Plot Fig 7")
        run_script(plot_dir / 'section_3/figure_08_missing_disinflation.py', "Plot Fig 8")
        run_script(plot_dir / 'section_3/figure_09_post_pandemic.py', "Plot Fig 9")
        run_script(plot_dir / 'section_3/figure_10_transmission_lags.py', "Plot Fig 10")
        run_script(plot_dir / 'section_3/figure_11_phillips_curve.py', "Plot Fig 11")
        run_script(plot_dir / 'section_3/figure_12_policy_asymmetry.py', "Plot Fig 12")
        run_script(plot_dir / 'section_3/figure_13_credibility_buffer.py', "Plot Fig 13")
        
        # Section 4
        run_script(plot_dir / 'section_4/figure_14_hyperinflation.py', "Plot Fig 14")
        run_script(plot_dir / 'section_4/figure_15_brazil.py', "Plot Fig 15")
        run_script(plot_dir / 'section_4/figure_16_long_memory.py', "Plot Fig 16")
        run_script(plot_dir / 'section_4/figure_17_japan.py', "Plot Fig 17")
        
        # Appendices
        run_script(plot_dir / 'appendix_B/figure_B1_reanchoring.py', "Plot Fig B1")
        run_script(plot_dir / 'appendix_E/figure_E1_mab_vs_bayesian.py', "Plot Fig E1")
        
    # ---------------------------------------------------------
    # 4. TABLES
    # ---------------------------------------------------------
    if not args.figures:
        print("\n" + "="*60)
        print("PHASE 4: TABLE GENERATION")
        print("="*60)
        
        table_dir = CODE_DIR / 'tables'
        run_script(table_dir / 'section_2/table_02_transitory_persistent.py', "Table 2: Transitory vs Persistent")
        run_script(table_dir / 'section_2/table_03_paradox_learning.py', "Table 3: Paradox of Learning")
        
        # Section 3 Tables
        run_script(table_dir / 'section_3/table_05_credibility_buffer.py', "Table 5: Credibility Buffer")
        run_script(table_dir / 'section_3/table_06_shock_gradual.py', "Table 6: Shock vs Gradual")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
