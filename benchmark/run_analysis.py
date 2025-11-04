#!/usr/bin/env python3
"""
Simple script to run quantization analysis directly.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from analyze_quantization_results import main

if __name__ == "__main__":
    # Set default arguments
    results_dir = "./results"
    output_dir = "./analysis"
    
    # Override sys.argv to pass arguments
    sys.argv = [
        "analyze_quantization_results.py",
        "--results-dir", results_dir,
        "--output-dir", output_dir
    ]
    
    print(f"Running analysis...")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 40)
    
    # Run the analysis
    exit_code = main()
    
    if exit_code == 0:
        print("\n" + "=" * 40)
        print("Analysis completed successfully!")
        print(f"Check {output_dir}/ for:")
        print("  - quant_summary.csv (the dataframe you wanted)")
        print("  - quantization_summary.csv (detailed stats)")
        print("  - quantization_comparison.csv (simplified comparison)")
        print("  - *.png files (visualizations)")
        print("  - quantization_benchmark_report.txt (full report)")
    else:
        print(f"\nAnalysis failed with exit code: {exit_code}")
    
    sys.exit(exit_code)