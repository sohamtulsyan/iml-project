"""
backtest/report.py — BacktestReport
6-panel tearsheet saved as PNGs + combined PDF.
"""
from __future__ import annotations
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except Exception:
    pass

try:
    from matplotlib.backends.backend_pdf import PdfPages
    _PDF_AVAILABLE = True
except Exception:
    _PDF_AVAILABLE = False

DPI = 200


class BacktestReport:
    def generate(
        self,
        source_name:    str,
        results_dir:    Path,
        benchmark_path: Path | None = None,
    ) -> None:
        results_dir = Path(results_dir)
        bt_dir      = results_dir / "backtest"
        res_path    = bt_dir / f"{source_name}_backtest_results.parquet"
        sum_path    = bt_dir / f"{source_name}_backtest_summary.json"

        if not res_path.exists():
            raise FileNotFoundError(f"Backtest results not found: {res_path}")

        df  = pd.read_parquet(res_path, engine="pyarrow")
        with open(sum_path) as f:
            summary = json.load(f)

        # Optional benchmark
        bench = None
        if benchmark_path and Path(benchmark_path).exists():
            try:
                bench = pd.read_csv(benchmark_path, parse_dates=["Month"])
            except Exception:
                bench = None

        figs = []

        # ── Plot 1: Cumulative returns ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df["Month"], df["cumulative_net"], label=source_name,
                color="#2ECC71", lw=2.5)
        if bench is not None:
            bench_cum = (1 + bench["benchmark_return"]).cumprod()
            ax.plot(bench["Month"], bench_cum, label="Benchmark",
                    color="#95A5A6", lw=1.5, alpha=0.8, ls="--")
        ax.axhline(1, color="black", ls="--", lw=1, alpha=0.4)
        ax.set_title(f"Cumulative Net Return — {source_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Cumulative Return (×)")
        ax.legend(); plt.tight_layout()
        path1 = bt_dir / f"{source_name}_plot_01_cumret.png"
        plt.savefig(path1, dpi=DPI, bbox_inches="tight"); plt.close()
        figs.append(path1)

        # ── Plot 2: Monthly returns bar ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        colors  = ["#2ECC71" if r >= 0 else "#E74C3C" for r in df["net_return"]]
        ax.bar(df["Month"], df["net_return"], color=colors, width=25, alpha=0.8)
        ax.axhline(df["net_return"].mean(), color="navy", ls="--", lw=1.5,
                   label=f"Mean {df['net_return'].mean():.2%}")
        ax.axhline(0, color="black", lw=1, alpha=0.4)
        ax.set_title(f"Monthly Net Returns — {source_name}", fontsize=14, fontweight="bold")
        ax.legend(); plt.tight_layout()
        path2 = bt_dir / f"{source_name}_plot_02_monthly.png"
        plt.savefig(path2, dpi=DPI, bbox_inches="tight"); plt.close()
        figs.append(path2)

        # ── Plot 3: Rolling 12m Sharpe ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        valid   = df.dropna(subset=["rolling_sharpe_12m"])
        ax.plot(valid["Month"], valid["rolling_sharpe_12m"],
                color="#9B59B6", lw=2)
        ax.axhline(0, color="black", ls="--", lw=1, alpha=0.4)
        ax.axhline(0.8, color="#F39C12", ls=":", lw=1.5, label="Sharpe=0.8 threshold")
        ax.set_title(f"Rolling 12-Month Sharpe — {source_name}", fontsize=14, fontweight="bold")
        ax.legend(); plt.tight_layout()
        path3 = bt_dir / f"{source_name}_plot_03_sharpe.png"
        plt.savefig(path3, dpi=DPI, bbox_inches="tight"); plt.close()
        figs.append(path3)

        # ── Plot 4: Drawdown ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        cum     = df["cumulative_net"].values
        peak    = np.maximum.accumulate(cum)
        dd      = (cum - peak) / peak
        ax.fill_between(df["Month"], dd, 0, color="#E74C3C", alpha=0.5)
        ax.plot(df["Month"], dd, color="#C0392B", lw=1.5)
        ax.set_title(f"Drawdown — {source_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Drawdown"); plt.tight_layout()
        path4 = bt_dir / f"{source_name}_plot_04_drawdown.png"
        plt.savefig(path4, dpi=DPI, bbox_inches="tight"); plt.close()
        figs.append(path4)

        # ── Plot 5: Long vs Short contribution by year ───────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        df["Year"] = pd.to_datetime(df["Month"]).dt.year
        annual = df.groupby("Year")[["long_leg_return","short_leg_return"]].sum()
        x      = np.arange(len(annual))
        w      = 0.35
        ax.bar(x - w/2, annual["long_leg_return"],  w, label="Long",  color="#2ECC71", alpha=0.8)
        ax.bar(x + w/2, annual["short_leg_return"], w, label="Short", color="#E74C3C", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(annual.index, rotation=45)
        ax.axhline(0, color="black", lw=1, alpha=0.4)
        ax.set_title(f"Annual Long/Short Contribution — {source_name}", fontsize=14, fontweight="bold")
        ax.legend(); plt.tight_layout()
        path5 = bt_dir / f"{source_name}_plot_05_attribution.png"
        plt.savefig(path5, dpi=DPI, bbox_inches="tight"); plt.close()
        figs.append(path5)

        # ── Plot 6: Summary table ─────────────────────────────────────────────
        rows = [
            ("Ann. Return",       f"{summary.get('annualized_return', 0):.2%}"),
            ("Ann. Volatility",   f"{summary.get('annualized_vol', 0):.2%}"),
            ("Sharpe Ratio",      f"{summary.get('sharpe_ratio', 0):.3f}"),
            ("Max Drawdown",      f"{summary.get('max_drawdown', 0):.2%}"),
            ("Calmar Ratio",      f"{summary.get('calmar_ratio', 0):.3f}"),
            ("Sortino Ratio",     f"{summary.get('sortino_ratio', 0):.3f}"),
            ("% Positive Months", f"{summary.get('pct_positive_months', 0):.1f}%"),
            ("Total Return",      f"{summary.get('total_return', 0):.2%}"),
            ("Long Leg (Ann.)",   f"{summary.get('long_return', 0):.2%}"),
            ("Short Leg (Ann.)",  f"{summary.get('short_return', 0):.2%}"),
            ("TC Drag (Ann.)",    f"{summary.get('tc_drag_annualized', 0):.2%}"),
            ("Months",            str(summary.get('n_months', 0))),
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2, 1.8)
        for (r, c), cell in t.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#F5F5F5")
            cell.set_edgecolor("white")
        ax.set_title(f"Performance Summary — {source_name}", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        path6 = bt_dir / f"{source_name}_plot_06_summary.png"
        plt.savefig(path6, dpi=DPI, bbox_inches="tight"); plt.close()
        figs.append(path6)

        # ── Combine to PDF ────────────────────────────────────────────────────
        if _PDF_AVAILABLE:
            pdf_path = bt_dir / f"{source_name}_tearsheet.pdf"
            with PdfPages(str(pdf_path)) as pdf:
                for fig_path in figs:
                    img = plt.imread(str(fig_path))
                    fig_p, ax_p = plt.subplots(figsize=(14, 8))
                    ax_p.imshow(img); ax_p.axis("off")
                    pdf.savefig(fig_p, bbox_inches="tight")
                    plt.close(fig_p)
            print(f"  ✓ {pdf_path}")

        print(f"[BacktestReport] {source_name} tearsheet saved to {bt_dir}/")
