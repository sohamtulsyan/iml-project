"""
run_pipeline.py
===============
Master CLI orchestrator for the unified equity prediction pipeline.

Usage:
    # Run all 7 models + ensemble + backtest
    python run_pipeline.py --data project_database.csv

    # Run specific models only
    python run_pipeline.py --data project_database.csv --models ridge cart lightgbm

    # Skip models already run (reload results)
    python run_pipeline.py --data project_database.csv --skip-existing

    # Run ensemble + backtest without rerunning models
    python run_pipeline.py --data project_database.csv --ensemble-only

    # Full pipeline with backtesting
    python run_pipeline.py --data project_database.csv --backtest

Pipeline stages:
    Layer 1 — Linear:   Ridge
    Layer 1 — Trees:    CART, LightGBM, Random Forest
    Layer 2 — Neural:   MLP, Transformer, CNN
    Layer 3 — Ensemble: SimpleAverage + MetaLearner (Ridge on OOF)
    Layer 4 — Backtest: PortfolioConstructor + BacktestEngine + BacktestReport
    Layer 4 — RL Stub:  RLAgent (placeholder)
"""
from __future__ import annotations
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from equity_pipeline.config import PipelineConfig
from equity_pipeline.shared.loader import load_data, build_target, lag_features
from equity_pipeline.shared.walk_forward import run_walk_forward
from equity_pipeline.shared.preprocessing import (
    make_linear_preprocessor, make_tree_preprocessor, make_sequence_preprocessor,
)
from equity_pipeline.shared.metrics import print_full_comparison
from equity_pipeline.shared.output import load_summary


def _make_all_models(cfg, models_to_run):
    """Instantiate selected models. Lazy imports keep CLI --help fast."""
    models = {}
    
    if "ridge" in models_to_run:
        from equity_pipeline.models.ridge import RidgeModel
        models["ridge"] = (RidgeModel(), make_linear_preprocessor)
    
    if "cart" in models_to_run:
        from equity_pipeline.models.cart import CARTModel
        models["cart"] = (CARTModel(), make_tree_preprocessor)
    
    if "lightgbm" in models_to_run:
        from equity_pipeline.models.lightgbm_model import LightGBMModel
        models["lightgbm"] = (LightGBMModel(seed=cfg.seed), make_tree_preprocessor)
    
    if "random_forest" in models_to_run:
        from equity_pipeline.models.random_forest import RandomForestModel
        models["random_forest"] = (RandomForestModel(seed=cfg.seed), make_tree_preprocessor)
    
    if "mlp" in models_to_run:
        from equity_pipeline.models.mlp import MLPModel
        models["mlp"] = (MLPModel(device_str=cfg.device, seed=cfg.seed, max_epochs=cfg.max_epochs,
                                  batch_size=cfg.batch_size, lr=cfg.lr),
                         make_sequence_preprocessor)
    
    if "transformer" in models_to_run:
        from equity_pipeline.models.transformer import TransformerModel
        models["transformer"] = (TransformerModel(device_str=cfg.device, seed=cfg.seed,
                                                   max_epochs=cfg.max_epochs,
                                                   batch_size=cfg.batch_size, lr=cfg.lr),
                                 make_sequence_preprocessor)
    
    if "cnn" in models_to_run:
        from equity_pipeline.models.cnn import CNNModel
        models["cnn"] = (CNNModel(device_str=cfg.device, seed=cfg.seed,
                                  max_epochs=cfg.max_epochs,
                                  batch_size=cfg.batch_size, lr=cfg.lr),
                         make_sequence_preprocessor)
    
    return models


def _parse_args():
    p = argparse.ArgumentParser(description="Unified Equity Prediction Pipeline")
    p.add_argument("--data",    default="project_database.csv",
                   help="Path to input CSV (default: project_database.csv)")
    p.add_argument("--results", default="results",
                   help="Results directory (default: results)")
    p.add_argument("--models",  nargs="+",
                   choices=["ridge","cart","lightgbm","random_forest",
                             "mlp","transformer","cnn","all"],
                   default=["all"], help="Which models to run")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip models that already have summary JSON")
    p.add_argument("--ensemble-only", action="store_true",
                   help="Skip model training, run ensemble + backtest only")
    p.add_argument("--backtest", action="store_true",
                   help="Run backtest after pipeline (requires all models done)")
    p.add_argument("--ensemble", choices=["none", "simple", "meta", "both"], default="both",
                   help="Ensemble strategy (default: both)")
    p.add_argument("--device", default="auto",
                   choices=["auto","cuda","mps","cpu"])
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--no-tune", action="store_true",
                   help="Skip hyperparameter tuning (use defaults)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args     = _parse_args()
    wall_t0  = time.time()

    cfg = PipelineConfig(
        data_path   = args.data,
        results_dir = args.results,
        device      = args.device,
        seed        = args.seed,
        verbose     = args.verbose,
    )

    results_dir = Path(cfg.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine which models to run ─────────────────────────────────────────
    ALL_MODELS = ["ridge","cart","lightgbm","random_forest","mlp","transformer","cnn"]
    if "all" in args.models:
        models_to_run = ALL_MODELS
    else:
        models_to_run = args.models

    # ── Load + preprocess data once ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  LOADING DATA")
    print("═" * 60)

    df = load_data(
        args.data, cfg.id_col, cfg.date_col, cfg.target_col, cfg.features
    )

    # Log-transform mktcap before anything else (global, not per-fold)
    for col in cfg.log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
            print(f"[Loader] {col} log-transformed globally")

    df = build_target(df, cfg.id_col, cfg.target_col)
    df = df.dropna(subset=list(cfg.features)).reset_index(drop=True)
    # Pre-cast features to float32 once for massive speedup in preprocessing (Issue E)
    df[list(cfg.features)] = df[list(cfg.features)].astype(np.float32)

    print(f"\n[Data] {df[cfg.date_col].nunique()} months | "
          f"{df[cfg.id_col].nunique():,} firms | "
          f"{len(df):,} firm-month observations\n")

    # ── Run models ────────────────────────────────────────────────────────────
    ic_summaries = {}

    if not args.ensemble_only:
        all_models = _make_all_models(cfg, models_to_run)

        for model_name in models_to_run:
            model, preprocessor_factory = all_models[model_name]
            model_dir = results_dir / model_name

            # Skip if already done
            if args.skip_existing and (model_dir / f"{model_name}_summary.json").exists():
                print(f"  [{model_name}] Skipping (summary exists)")
                ic_summaries[model_name] = load_summary(model_name, results_dir)
                continue

            print(f"\n{'═'*60}")
            print(f"  RUNNING {model_name.upper()}")
            print(f"{'═'*60}")

            compute_oof = args.ensemble in ("meta", "both")

            t0 = time.time()
            summary = run_walk_forward(
                df           = df,
                model        = model,
                cfg          = cfg,
                preprocessor_factory = preprocessor_factory,
                output_dir   = results_dir,
                compute_oof  = compute_oof,
                tune_first_fold = not args.no_tune,
            )
            elapsed = (time.time() - t0) / 60
            print(f"\n  [{model_name}] Completed in {elapsed:.1f} min")

            if summary:
                ic_summaries[model_name] = summary

    else:
        # Load existing summaries
        for mn in ALL_MODELS:
            s = load_summary(mn, results_dir)
            if s:
                ic_summaries[mn] = s

    # ── Ensemble ──────────────────────────────────────────────────────────────
    finished_models = [mn for mn in ALL_MODELS
                       if (results_dir / mn / f"{mn}_test_predictions.parquet").exists()]

    ens_ic_summaries = {}

    if len(finished_models) >= 2:
        from equity_pipeline.ensemble.simple_average import SimpleAverageEnsemble
        from equity_pipeline.ensemble.meta_learner   import MetaLearnerEnsemble

        print(f"\n{'═'*60}")
        print(f"  ENSEMBLE  ({len(finished_models)} base models)")
        print(f"{'═'*60}")

        # Simple average (always runs if ≥ 2 test predictions exist)
        try:
            avg = SimpleAverageEnsemble(finished_models, results_dir / "ensemble", cfg)
            ens_ic_summaries["simple_average"] = avg.run()
        except Exception as e:
            print(f"  [SimpleAverage] Error: {e}")

        # MetaLearner (requires OOF predictions for all models)
        oof_ready = [mn for mn in finished_models
                     if (results_dir / mn / f"{mn}_oof_predictions.parquet").exists()]
        if len(oof_ready) >= 2:
            try:
                ml  = MetaLearnerEnsemble(oof_ready, results_dir / "ensemble", cfg)
                ens_ic_summaries["meta_learner"] = ml.run()
            except Exception as e:
                print(f"  [MetaLearner] Error: {e}")
        else:
            print(f"  [MetaLearner] Need OOF predictions for ≥ 2 models "
                  f"(found: {oof_ready})")

    # ── Backtest ──────────────────────────────────────────────────────────────
    if args.backtest:
        from equity_pipeline.backtest.portfolio import PortfolioConstructor
        from equity_pipeline.backtest.engine    import BacktestEngine
        from equity_pipeline.backtest.report    import BacktestReport

        print(f"\n{'═'*60}")
        print(f"  BACKTEST")
        print(f"{'═'*60}")

        pc  = PortfolioConstructor(cfg)
        eng = BacktestEngine(cfg)
        rpt = BacktestReport()

        bt_sources = finished_models + [s for s in ["simple_average","meta_learner"]
                     if (results_dir / "ensemble" / f"{s}_test_predictions.parquet").exists()]

        port_summaries = {}
        for source in bt_sources:
            if source in ["simple_average","meta_learner"]:
                pred_path = results_dir / "ensemble" / f"{source}_test_predictions.parquet"
                bt_dir    = results_dir / "ensemble" / "backtest"
            else:
                pred_path = results_dir / source / f"{source}_test_predictions.parquet"
                bt_dir    = results_dir / "backtest"

            if not pred_path.exists():
                continue
            try:
                pos_path = pc.run(pred_path, source, results_dir / "backtest")
                port_summaries[source] = eng.run(pos_path, source)
                rpt.generate(source, results_dir)
            except Exception as e:
                print(f"  [Backtest/{source}] Error: {e}")

    # ── Final comparison table ────────────────────────────────────────────────
    all_ic = {**ic_summaries, **ens_ic_summaries}
    if all_ic:
        df_cmp = print_full_comparison(all_ic)
        cmp_path = results_dir / "full_comparison.csv"
        df_cmp.to_csv(cmp_path, index=False)
        print(f"  ✓ {cmp_path}")

    total_min = (time.time() - wall_t0) / 60
    print(f"\n[Pipeline] Total time: {total_min:.1f} min")
    print("[Pipeline] Done ✓")


if __name__ == "__main__":
    main()
