#!/usr/bin/env python
"""
================================================================================
  MLP UNIFIED RUNNER — TRAIN + VISUALIZE + COMPARE
  One-command execution for complete MLP pipeline
================================================================================

Usage:
    python run_mlp_pipeline.py [--train] [--visualize] [--all]

Options:
    --train      Run training only
    --visualize  Run visualization only
    --all        Run training + visualization (default)

Examples:
    python run_mlp_pipeline.py                    # Train + visualize
    python run_mlp_pipeline.py --train            # Train only
    python run_mlp_pipeline.py --visualize        # Visualize only (requires training first)

================================================================================
"""

import sys
import subprocess
from pathlib import Path
import argparse

def run_command(script_name: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print("\n" + "=" * 80)
    print(f"  {description.upper()}")
    print("=" * 80 + "\n")
    
    script_path = Path(__file__).parent / script_name
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=Path(__file__).parent
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_path}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="MLP Pipeline Runner — Train + Visualize + Compare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Train + visualize (default)
  %(prog)s --train            # Train only
  %(prog)s --visualize        # Visualize only (requires training first)
        """
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training only'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Run visualization only'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run training + visualization (default)'
    )
    
    args = parser.parse_args()
    
    # Determine what to run
    should_train = args.train or args.all or (not args.train and not args.visualize and not args.all)
    should_visualize = args.visualize or args.all or (not args.train and not args.visualize and not args.all)
    
    if not should_train and not should_visualize:
        parser.print_help()
        return 1
    
    # Run pipeline
    success = True
    
    if should_train:
        success = run_command('train_mlp.py', 'Step 1: Train MLP with Walk-Forward Validation') and success
    
    if should_visualize:
        if not should_train:
            print("\n[Visualization] Checking if training results exist...")
            ic_csv = Path(__file__).parent / 'mlp_ic_results.csv'
            if not ic_csv.exists():
                print(f"✗ Training results not found: {ic_csv}")
                print("  Please run training first: python run_mlp_pipeline.py --train")
                return 1
        
        success = run_command('visualize_mlp_results.py', 'Step 2: Generate Comparison Visualizations') and success
    
    # Summary
    print("\n" + "=" * 80)
    if success:
        print("  ✓ MLP PIPELINE COMPLETE")
        print("\n  Results:")
        print("  • mlp_ic_results.csv")
        print("  • mlp_hyperparams.csv")
        print("  • mlp_training_times.csv")
        print("  • visualizations/*.png")
    else:
        print("  ✗ MLP PIPELINE ENCOUNTERED ERRORS")
    print("=" * 80 + "\n")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
