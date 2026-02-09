#!/usr/bin/env python3
"""
ParEval Metrics Report Generator

Generates comprehensive metrics from ParEval benchmark results.
Based on metrics from Nichols et al. "Can Large Language Models Write Parallel Code?"

Usage:
    python pareval_report.py <outputs_folder> [--output report.txt]
    
Example:
    python pareval_report.py ../deepseek-outputs
    python pareval_report.py ../deepseek-outputs --output deepseek_report.txt
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json


def load_metrics(folder: Path) -> pd.DataFrame:
    """Load and combine all metrics CSV files from the folder."""
    metrics_files = list(folder.glob("metrics_*.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics_*.csv files found in {folder}")
    
    dfs = []
    for f in metrics_files:
        if f.name == "metrics_combined.csv":
            continue
        df = pd.read_csv(f)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def load_results(folder: Path) -> pd.DataFrame:
    """Load and combine all results CSV files from the folder."""
    results_files = list(folder.glob("results_*.csv"))
    if not results_files:
        return None
    
    dfs = []
    for f in results_files:
        if f.name == "results_combined.csv":
            continue
        df = pd.read_csv(f)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def print_header(text: str, char: str = "=", width: int = 75):
    print()
    print(char * width)
    print(text)
    print(char * width)


def print_section(text: str, char: str = "-", width: int = 75):
    print()
    print(text)
    print(char * width)


def generate_report(folder: Path, output_file: Path = None):
    """Generate comprehensive ParEval metrics report."""
    
    # Redirect output if specified
    original_stdout = sys.stdout
    if output_file:
        sys.stdout = open(output_file, "w", encoding="utf-8")
    
    try:
        metrics = load_metrics(folder)
        results = load_results(folder)
        
        model_name = folder.name
        problem_types = sorted(metrics["problem type"].unique())
        exec_models = sorted(metrics["execution model"].unique())
        
        # =====================================================================
        # HEADER
        # =====================================================================
        print_header(f"ParEval Benchmark Report: {model_name}")
        print(f"Output folder: {folder.absolute()}")
        print(f"Problem types: {len(problem_types)} ({', '.join(problem_types)})")
        print(f"Execution models: {', '.join(exec_models)}")
        print(f"Total metric rows: {len(metrics)}")
        if results is not None:
            print(f"Total result rows: {len(results)}")
        
        # =====================================================================
        # 1. OVERALL SUMMARY
        # =====================================================================
        print_header("1. OVERALL SUMMARY")
        
        summary_data = []
        for em in exec_models:
            em_df = metrics[metrics["execution model"] == em]
            summary_data.append({
                "Execution Model": em.upper(),
                "build@1": f"{em_df['build@1'].mean():.1%}",
                "pass@1": f"{em_df['pass@1'].mean():.1%}",
                "pass@5": f"{em_df['pass@5'].mean():.1%}",
                "pass@10": f"{em_df['pass@10'].mean():.1%}",
                "pass@20": f"{em_df['pass@20'].mean():.1%}",
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # =====================================================================
        # 2. CORRECTNESS BY PROBLEM TYPE (pass@1)
        # =====================================================================
        print_header("2. CORRECTNESS BY PROBLEM TYPE (pass@1)")
        
        pass1_pivot = metrics.pivot_table(
            index="problem type",
            columns="execution model",
            values="pass@1",
            aggfunc="first"
        )
        
        # Reorder columns if they exist
        cols = [c for c in ["serial", "omp", "cuda", "mpi", "hip", "kokkos"] if c in pass1_pivot.columns]
        pass1_pivot = pass1_pivot[cols]
        
        # Format as percentages
        pass1_display = pass1_pivot.apply(lambda x: x.map(lambda v: f"{v:.1%}" if pd.notna(v) else "-"))
        print(pass1_display.to_string())
        
        # =====================================================================
        # 3. SERIAL VS PARALLEL COMPARISON
        # =====================================================================
        if "serial" in exec_models and "omp" in exec_models:
            print_header("3. SERIAL VS PARALLEL (OMP) COMPARISON")
            
            serial_df = metrics[metrics["execution model"] == "serial"].set_index("problem type")
            omp_df = metrics[metrics["execution model"] == "omp"].set_index("problem type")
            
            comparison = pd.DataFrame({
                "Serial pass@1": serial_df["pass@1"],
                "OMP pass@1": omp_df["pass@1"],
                "Gap": omp_df["pass@1"] - serial_df["pass@1"],
                "OMP speedup_max": omp_df["speedup_max@1"],
            })
            comparison = comparison.sort_values("Gap", ascending=False)
            
            # Format
            comp_display = comparison.copy()
            comp_display["Serial pass@1"] = comp_display["Serial pass@1"].map(lambda v: f"{v:.1%}")
            comp_display["OMP pass@1"] = comp_display["OMP pass@1"].map(lambda v: f"{v:.1%}")
            comp_display["Gap"] = comp_display["Gap"].map(lambda v: f"{v:+.1%}")
            comp_display["OMP speedup_max"] = comp_display["OMP speedup_max"].map(lambda v: f"{v:.2f}x")
            print(comp_display.to_string())
            
            print_section("Interpretation:")
            omp_better = (comparison["Gap"] > 0).sum()
            serial_better = (comparison["Gap"] < 0).sum()
            print(f"  - OMP outperforms serial on {omp_better}/{len(comparison)} problem types")
            print(f"  - Serial outperforms OMP on {serial_better}/{len(comparison)} problem types")
            print(f"  - Average gap: {comparison['Gap'].mean():+.1%}")
        
        # =====================================================================
        # 4. SPEEDUP ANALYSIS
        # =====================================================================
        print_header("4. SPEEDUP ANALYSIS (speedup_max@1)")
        print("speedup > 1 means parallel code is FASTER than serial baseline")
        print("speedup < 1 means parallel code is SLOWER (parallelization overhead)")
        
        speedup_pivot = metrics.pivot_table(
            index="problem type",
            columns="execution model",
            values="speedup_max@1",
            aggfunc="first"
        )
        cols = [c for c in ["serial", "omp", "cuda"] if c in speedup_pivot.columns]
        speedup_pivot = speedup_pivot[cols]
        
        speedup_display = speedup_pivot.apply(lambda x: x.map(lambda v: f"{v:.2f}x" if pd.notna(v) and v > 0 else "-"))
        print(speedup_display.to_string())
        
        if "omp" in exec_models:
            omp_speedups = metrics[metrics["execution model"] == "omp"]
            passing_omp = omp_speedups[omp_speedups["pass@1"] > 0]
            achieves_speedup = (passing_omp["speedup_max@1"] > 1).sum()
            total_passing = len(passing_omp)
            print_section("OMP Speedup Summary:")
            print(f"  - Problem types where OMP passes: {total_passing}")
            print(f"  - Of those, achieves speedup > 1: {achieves_speedup} ({achieves_speedup/total_passing:.0%})")
            
            slow_problems = passing_omp[passing_omp["speedup_max@1"] < 1]["problem type"].tolist()
            if slow_problems:
                print(f"  - OMP SLOWER than serial on: {', '.join(slow_problems)}")
        
        # =====================================================================
        # 5. BUILD VS PASS ANALYSIS
        # =====================================================================
        print_header("5. BUILD VS PASS ANALYSIS")
        print("Shows how much code compiles but fails tests (correctness issues)")
        
        build_pass = []
        for em in exec_models:
            em_df = metrics[metrics["execution model"] == em]
            build_rate = em_df["build@1"].mean()
            pass_rate = em_df["pass@1"].mean()
            build_pass.append({
                "Model": em.upper(),
                "Build Rate": f"{build_rate:.1%}",
                "Pass Rate": f"{pass_rate:.1%}",
                "Builds but Fails": f"{build_rate - pass_rate:.1%}",
                "Pass/Build Ratio": f"{pass_rate/build_rate:.1%}" if build_rate > 0 else "-"
            })
        
        print(pd.DataFrame(build_pass).to_string(index=False))
        
        # =====================================================================
        # 6. CUDA ANALYSIS (if present)
        # =====================================================================
        if "cuda" in exec_models:
            print_header("6. CUDA ANALYSIS")
            
            cuda_df = metrics[metrics["execution model"] == "cuda"]
            print(f"CUDA builds: {cuda_df['build@1'].mean():.1%}")
            print(f"CUDA passes: {cuda_df['pass@1'].mean():.1%}")
            print(f"CUDA improvement with more samples: pass@1={cuda_df['pass@1'].mean():.1%} -> pass@20={cuda_df['pass@20'].mean():.1%}")
            
            if cuda_df["pass@1"].mean() < cuda_df["build@1"].mean() * 0.5:
                print("\n[!] CUDA code mostly compiles but produces wrong results!")
                print("    This is a common LLM failure mode for GPU code.")
        
        # =====================================================================
        # 7. PROBLEM TYPE DIFFICULTY RANKING
        # =====================================================================
        print_header("7. PROBLEM TYPE DIFFICULTY RANKING")
        print("Ranked by average pass@1 across all execution models (hardest first)")
        
        difficulty = metrics.groupby("problem type")["pass@1"].mean().sort_values()
        print()
        for i, (pt, rate) in enumerate(difficulty.items(), 1):
            bar = "#" * int(rate * 30)
            print(f"  {i:2}. {pt:15} {rate:5.1%} {bar}")
        
        # =====================================================================
        # 8. EFFICIENCY ANALYSIS
        # =====================================================================
        if "omp" in exec_models:
            print_header("8. PARALLEL EFFICIENCY (efficiency_max@1)")
            print("Efficiency = speedup / num_threads (ideal = 1.0)")
            print("Values > 1 indicate super-linear speedup (cache effects, etc.)")
            
            eff_df = metrics[metrics["execution model"] == "omp"][["problem type", "efficiency_max@1"]]
            eff_df = eff_df.set_index("problem type").sort_values("efficiency_max@1", ascending=False)
            
            for pt, row in eff_df.iterrows():
                eff = row["efficiency_max@1"]
                if eff > 0:
                    bar = "#" * min(int(eff * 20), 50)
                    print(f"  {pt:15} {eff:5.2f} {bar}")
        
        # =====================================================================
        # 9. PASS@K IMPROVEMENT CURVE
        # =====================================================================
        print_header("9. PASS@K IMPROVEMENT CURVE")
        print("Shows how pass rate improves with more samples")
        
        for em in exec_models:
            em_df = metrics[metrics["execution model"] == em]
            print(f"\n{em.upper()}:")
            for k in [1, 5, 10, 20]:
                rate = em_df[f"pass@{k}"].mean()
                bar = "#" * int(rate * 40)
                print(f"  pass@{k:2}: {rate:5.1%} {bar}")
        
        # =====================================================================
        # 10. RAW DATA EXPORT INFO
        # =====================================================================
        print_header("10. OUTPUT FILES")
        
        # Save combined metrics
        combined_metrics_path = folder / "metrics_combined.csv"
        metrics.to_csv(combined_metrics_path, index=False)
        print(f"Combined metrics saved to: {combined_metrics_path}")
        
        if results is not None:
            combined_results_path = folder / "results_combined.csv"
            results.to_csv(combined_results_path, index=False)
            print(f"Combined results saved to: {combined_results_path}")
        
        # =====================================================================
        # FOOTER
        # =====================================================================
        print()
        print("=" * 75)
        print("END OF REPORT")
        print("=" * 75)
        
    finally:
        if output_file:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ParEval benchmark metrics report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("outputs_folder", type=Path, help="Path to folder containing metrics CSV files")
    parser.add_argument("-o", "--output", type=Path, help="Output file for report (default: print to stdout)")
    
    args = parser.parse_args()
    
    if not args.outputs_folder.exists():
        print(f"Error: Folder not found: {args.outputs_folder}", file=sys.stderr)
        sys.exit(1)
    
    generate_report(args.outputs_folder, args.output)


if __name__ == "__main__":
    main()
