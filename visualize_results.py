"""
Script 6 — Visualize Regression Results
=========================================
Reads matched_panel.csv and sensitivity_results.csv produced by
Scripts 4 and 5 and generates a set of publication-ready charts.

Charts produced:
  1.  Mean home value by group and period (bar chart)
  2.  Mean rent by group and period (bar chart)
  3.  DiD effect size with 95% CI — home value and rent (horizontal bar)
  4.  Buffer sensitivity — β buildout×2023 across 2/4/6/8km (line chart)
  5.  Home value change 2000→2023 by group (bar)
  6.  Rent change 2000→2023 by group (bar)
  7.  County-level comparison (grouped bar)
  8.  Baseline-controlled gap analysis — percent gap by period (table + bar)

All charts saved to: outputs/figures/

Install: pip install pandas matplotlib numpy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, MultipleLocator
import warnings
warnings.filterwarnings("ignore")

# ── SETTINGS ──────────────────────────────────────────────────────────────────
PANEL_CSV       = "data/matched_panel.csv"
SENSITIVITY_CSV = "data/sensitivity_results.csv"
OUTPUT_DIR      = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "control":  "#6B7280",
    "buildout": "#0F766E",
    "accent":   "#F59E0B",
    "negative": "#DC2626",
    "positive": "#0F766E",
    "ci":       "#CCFBF1",
}

YEARS       = [2000, 2015, 2023]
YEAR_LABELS = ["2000\n(Baseline)", "2015\n(Mid-period)", "2023\n(End-period)"]

# Hardcoded TWFE results from Script 5 primary specification (8km)
TWFE_RESULTS = {
    "med_home_value": {
        "buildout_x_post_2015": {"coef": -1171.45,   "se": 26366.71, "p": 0.965},
        "buildout_x_post_2023": {"coef": 123306.97,  "se": 46097.27, "p": 0.008},
    },
    "med_rent": {
        "buildout_x_post_2015": {"coef": 48.19,  "se": 43.44, "p": 0.268},
        "buildout_x_post_2023": {"coef": 133.90, "se": 67.16, "p": 0.047},
    },
}

DID_RESULTS = {
    "med_home_value": {
        "buildout_x_post_2015": {"coef": -4554.77,   "se": 21886.62, "p": 0.835},
        "buildout_x_post_2023": {"coef": 119639.98,  "se": 37929.68, "p": 0.002},
    },
    "med_rent": {
        "buildout_x_post_2015": {"coef": 47.82,  "se": 35.71, "p": 0.181},
        "buildout_x_post_2023": {"coef": 134.59, "se": 54.94, "p": 0.014},
    },
}


# ── FORMATTERS ────────────────────────────────────────────────────────────────

def dollar_fmt(x, pos):
    """Home values — large numbers compressed to $XXXk or $X.XM"""
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    elif abs(x) >= 1_000:
        return f"${x/1_000:.0f}k"
    return f"${x:.0f}"


def rent_fmt(x, pos):
    """Rent — full dollar display with comma separator, no k compression"""
    if abs(x) >= 1_000:
        return f"${x:,.0f}"
    return f"${x:.0f}"


def pct_fmt(x, pos):
    """Percentage formatter"""
    return f"{x:.1f}%"


def sig_label(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return "n.s."


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#4B5563", labelsize=9)
    ax.yaxis.grid(True, color="#F3F4F6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ── CHART 1 & 2: MEAN OUTCOME BY GROUP AND PERIOD ────────────────────────────

def chart_mean_by_group(panel: pd.DataFrame):
    print("\nGenerating Charts 1 & 2: Mean outcome by group and period...")

    specs = [
        ("med_home_value",
         "Median Home Value by Group and Period (8km buffer)",
         dollar_fmt, None, "01_home_value_by_group.png"),
        ("med_rent",
         "Median Gross Rent by Group and Period (8km buffer)",
         rent_fmt, 250, "02_rent_by_group.png"),
    ]

    for outcome, title, fmt, tick_interval, fname in specs:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        groups = ["control", "buildout"]
        x      = np.arange(len(YEARS))
        width  = 0.35

        for i, group in enumerate(groups):
            means = []
            cis   = []
            for year in YEARS:
                vals = panel[
                    (panel["year"] == year) &
                    (panel["group"] == group)
                ][outcome].dropna()
                means.append(vals.mean())
                cis.append(1.96 * vals.std() / np.sqrt(len(vals))
                            if len(vals) > 1 else 0)

            offset = (i - 0.5) * width
            bars   = ax.bar(
                x + offset, means, width,
                color=COLORS[group], alpha=0.88, zorder=3,
                label=group.capitalize(),
                yerr=cis, capsize=4,
                error_kw={"ecolor": "#374151", "linewidth": 1.2}
            )

            for bar, mean in zip(bars, means):
                if mean > 0:
                    label = (f"${mean/1000:.0f}k" if mean >= 1000
                             else f"${mean:,.0f}")
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        label, ha="center", va="bottom",
                        fontsize=8, color="#374151", fontweight="bold"
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(YEAR_LABELS)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        if tick_interval:
            ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
        style_ax(ax, title, ylabel=outcome.replace("_", " ").title())
        ax.legend(frameon=False, fontsize=10)

        fig.text(0.5, -0.02,
                 "Note: Matched PSM sample (8km buffer). "
                 "Error bars show 95% CI of mean.",
                 ha="center", fontsize=8, color="#6B7280", style="italic")
        save_fig(fig, fname)


# ── CHART 3: EFFECT SIZE WITH 95% CI ─────────────────────────────────────────

def chart_effect_size(panel: pd.DataFrame):
    print("\nGenerating Chart 3: Effect size with 95% CI...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    specs = [
        ("med_home_value",
         "Effect on Median Home Value\n(2023, relative to 2000 baseline)",
         dollar_fmt, None, axes[0]),
        ("med_rent",
         "Effect on Median Gross Rent\n(2023, relative to 2000 baseline)",
         rent_fmt, 50, axes[1]),
    ]

    for outcome, title, fmt, tick_interval, ax in specs:
        models = [
            ("DiD",  DID_RESULTS[outcome]["buildout_x_post_2023"]),
            ("TWFE", TWFE_RESULTS[outcome]["buildout_x_post_2023"]),
        ]

        y_pos  = np.arange(len(models))
        coefs  = [m[1]["coef"] for m in models]
        ses    = [m[1]["se"]   for m in models]
        ps     = [m[1]["p"]    for m in models]
        ci95   = [1.96 * se for se in ses]
        colors = [COLORS["positive"] if c > 0 else COLORS["negative"]
                  for c in coefs]

        ax.barh(y_pos, coefs, height=0.45,
                color=colors, alpha=0.85, zorder=3)
        ax.errorbar(coefs, y_pos, xerr=ci95,
                    fmt="none", color="#1F2937",
                    capsize=5, linewidth=1.5, zorder=4)
        ax.axvline(0, color="#374151", linewidth=1.2,
                   linestyle="--", alpha=0.5)

        for i, (coef, ci, p) in enumerate(zip(coefs, ci95, ps)):
            label = sig_label(p)
            x_pos = coef + ci + abs(max(coefs)) * 0.05
            ax.text(x_pos, i, label, va="center",
                    fontsize=11, fontweight="bold",
                    color="#0F766E" if p < 0.05 else "#9CA3AF")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([m[0] for m in models], fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        if tick_interval:
            ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        style_ax(ax, title, xlabel="Coefficient (β)")

    fig.suptitle(
        "Buildout DC Effect on Housing Costs — DiD vs TWFE (8km Buffer, 2023)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.text(0.5, -0.04,
             "* p<0.1   ** p<0.05   *** p<0.01   n.s. not significant\n"
             "Error bars show 95% confidence intervals. "
             "Clustered SE at tract level.",
             ha="center", fontsize=8, color="#6B7280", style="italic")
    plt.tight_layout()
    save_fig(fig, "03_effect_size_ci.png")


# ── CHART 4: BUFFER SENSITIVITY ───────────────────────────────────────────────

def chart_buffer_sensitivity(sensitivity: pd.DataFrame):
    print("\nGenerating Chart 4: Buffer sensitivity analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    specs = [
        ("med_home_value",
         "Home Value: β(buildout×2023) by Buffer Size",
         dollar_fmt, None, axes[0]),
        ("med_rent",
         "Rent: β(buildout×2023) by Buffer Size",
         rent_fmt, 100, axes[1]),
    ]

    for outcome, title, fmt, tick_interval, ax in specs:
        sub = sensitivity[
            (sensitivity["outcome"] == outcome) &
            (sensitivity["term"] == "buildout_x_post_2023")
        ].sort_values("buffer_km")

        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        x     = sub["buffer_km"].values
        coefs = sub["coef"].values
        ses   = sub["se"].values
        ps    = sub["p"].values
        ci95  = 1.96 * ses

        ax.fill_between(x, coefs - ci95, coefs + ci95,
                        alpha=0.15, color=COLORS["buildout"], zorder=2)
        ax.plot(x, coefs, color=COLORS["buildout"],
                linewidth=2.5, marker="o", markersize=8,
                markerfacecolor="white", markeredgewidth=2.5, zorder=3)
        ax.axhline(0, color="#374151", linewidth=1,
                   linestyle="--", alpha=0.5)

        for xi, coef, ci, p in zip(x, coefs, ci95, ps):
            label = sig_label(p)
            color = COLORS["positive"] if p < 0.05 else "#9CA3AF"
            offset = abs(max(coefs)) * 0.06
            ax.text(xi, coef + ci + offset, label,
                    ha="center", fontsize=10,
                    fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}km" for b in x])
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        if tick_interval:
            ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
        style_ax(ax, title,
                 xlabel="Buffer radius",
                 ylabel="β coefficient (buildout × post-2023)")

    fig.suptitle(
        "Buffer Sensitivity Analysis — Effect on Housing Costs at 2/4/6/8km",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.text(0.5, -0.04,
             "* p<0.1   ** p<0.05   *** p<0.01   Shaded band = 95% CI\n"
             "Direction consistent positive across all buffer sizes.",
             ha="center", fontsize=8, color="#6B7280", style="italic")
    plt.tight_layout()
    save_fig(fig, "04_buffer_sensitivity.png")


# ── CHART 5 & 6: ABSOLUTE CHANGE 2000→2023 BY GROUP ──────────────────────────

def chart_absolute_change(panel: pd.DataFrame):
    print("\nGenerating Charts 5 & 6: Absolute change 2000→2023 by group...")

    specs = [
        ("med_home_value",
         "Change in Median Home Value 2000→2023 by Treatment Group",
         dollar_fmt, None,
         "05_home_value_change.png"),
        ("med_rent",
         "Change in Median Gross Rent 2000→2023 by Treatment Group",
         rent_fmt, 200,
         "06_rent_change.png"),
    ]

    for outcome, title, fmt, tick_interval, fname in specs:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        groups     = ["control", "buildout"]
        changes    = []
        base_means = []
        errors     = []

        for group in groups:
            base = panel[
                (panel["year"] == 2000) & (panel["group"] == group)
            ][outcome].dropna()
            end  = panel[
                (panel["year"] == 2023) & (panel["group"] == group)
            ][outcome].dropna()

            changes.append(end.mean() - base.mean())
            base_means.append(base.mean())
            se = np.sqrt((base.std()**2 / len(base)) +
                          (end.std()**2  / len(end)))
            errors.append(1.96 * se)

        x      = np.arange(len(groups))
        colors = [COLORS["control"], COLORS["buildout"]]

        ax.bar(x, changes, width=0.5,
               color=colors, alpha=0.88, zorder=3,
               yerr=errors, capsize=6,
               error_kw={"ecolor": "#374151", "linewidth": 1.5})

        for i, (change, err) in enumerate(zip(changes, errors)):
            label = (f"${change/1000:.0f}k" if abs(change) >= 1000
                     else f"${change:,.0f}")
            ypos  = change + err * 0.2
            ax.text(x[i], ypos, label,
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#1F2937")

        diff  = changes[1] - changes[0]
        dlabel = (f"Difference: ${diff/1000:.0f}k"
                  if abs(diff) >= 1000 else f"Difference: ${diff:,.0f}")
        ax.annotate(
            dlabel,
            xy=(0.5, max(changes) * 0.82),
            xycoords=("axes fraction", "data"),
            ha="center", fontsize=10,
            color=COLORS["buildout"] if diff > 0 else COLORS["negative"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=COLORS["ci"],
                      edgecolor=COLORS["buildout"], alpha=0.8)
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Control\n(no DC nearby)", "Buildout\n(DC within 8km)"],
            fontsize=11
        )
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        if tick_interval:
            ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
        style_ax(ax, title, ylabel="Change in value (2000→2023)")
        ax.axhline(0, color="#374151", linewidth=1,
                   linestyle="--", alpha=0.4)

        base_label = (
            f"Baseline (2000): "
            f"Control = ${base_means[0]/1000:.0f}k, "
            f"Buildout = ${base_means[1]/1000:.0f}k."
            if base_means[0] >= 1000
            else
            f"Baseline (2000): "
            f"Control = ${base_means[0]:,.0f}, "
            f"Buildout = ${base_means[1]:,.0f}."
        )
        fig.text(0.5, -0.02,
                 f"{base_label}  Matched PSM sample. Error bars show 95% CI.",
                 ha="center", fontsize=8, color="#6B7280", style="italic")
        save_fig(fig, fname)


# ── CHART 7: COUNTY-LEVEL COMPARISON ─────────────────────────────────────────

def chart_county_comparison(panel: pd.DataFrame):
    print("\nGenerating Chart 7: County-level comparison...")

    counties = [
        "Sacramento County",
        "Los Angeles County",
        "Alameda County",
        "San Francisco County",
    ]
    short_names = ["Sacramento", "Los Angeles", "Alameda", "San Francisco"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    specs = [
        ("med_home_value", "Home Value Change 2000→2023 by County",
         dollar_fmt, None, axes[0]),
        ("med_rent",       "Rent Change 2000→2023 by County",
         rent_fmt, 200, axes[1]),
    ]

    for outcome, title, fmt, tick_interval, ax in specs:
        x     = np.arange(len(counties))
        width = 0.35

        for i, group in enumerate(["control", "buildout"]):
            changes = []
            for county in counties:
                base = panel[
                    (panel["year"] == 2000) &
                    (panel["group"] == group) &
                    (panel["county_name"] == county)
                ][outcome].dropna()
                end = panel[
                    (panel["year"] == 2023) &
                    (panel["group"] == group) &
                    (panel["county_name"] == county)
                ][outcome].dropna()
                changes.append(
                    end.mean() - base.mean()
                    if len(base) > 0 and len(end) > 0 else 0
                )

            offset = (i - 0.5) * width
            ax.bar(x + offset, changes, width,
                   color=COLORS[group], alpha=0.88,
                   zorder=3, label=group.capitalize())

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        if tick_interval:
            ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.axhline(0, color="#374151", linewidth=1,
                   linestyle="--", alpha=0.4)
        style_ax(ax, title, ylabel="Change (2000→2023)")
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Housing Cost Changes by County and Treatment Group (2000→2023)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.text(0.5, -0.02,
             "Matched PSM sample. Raw means — not regression-adjusted.",
             ha="center", fontsize=8, color="#6B7280", style="italic")
    plt.tight_layout()
    save_fig(fig, "07_county_comparison.png")


# ── CHART 8: BASELINE-CONTROLLED GAP ANALYSIS ────────────────────────────────
# Shows the gap between buildout and control as a % of control mean
# at each period, then subtracts the baseline gap to isolate the
# net effect above and beyond the pre-existing difference.

def chart_gap_analysis(panel: pd.DataFrame):
    print("\nGenerating Chart 8: Baseline-controlled gap analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    specs = [
        ("med_home_value", "Home Value: Buildout–Control Gap by Period",
         dollar_fmt, None, axes[0]),
        ("med_rent",       "Rent: Buildout–Control Gap by Period",
         rent_fmt, 50, axes[1]),
    ]

    # Print gap table to console
    print(f"\n{'='*70}")
    print("BASELINE-CONTROLLED GAP TABLE")
    print(f"{'='*70}")

    for outcome, title, fmt, tick_interval, ax in specs:
        print(f"\n  {outcome.upper()}")
        print(f"  {'Period':<10} {'Control':>12} {'Buildout':>12} "
              f"{'Gap ($)':>12} {'Gap (%)':>10} {'Net vs 2000':>14}")
        print("  " + "─" * 62)

        means    = {}
        gaps_abs = []
        gaps_pct = []

        for year in YEARS:
            ctrl = panel[
                (panel["year"] == year) & (panel["group"] == "control")
            ][outcome].dropna().mean()
            bld  = panel[
                (panel["year"] == year) & (panel["group"] == "buildout")
            ][outcome].dropna().mean()
            means[year] = {"control": ctrl, "buildout": bld}
            gap_abs     = bld - ctrl
            gap_pct     = (gap_abs / ctrl * 100) if ctrl > 0 else np.nan
            gaps_abs.append(gap_abs)
            gaps_pct.append(gap_pct)

        baseline_pct = gaps_pct[0]
        for i, year in enumerate(YEARS):
            net     = gaps_pct[i] - baseline_pct
            net_str = f"{net:+.1f}pp" if i > 0 else "—"
            ctrl_v  = means[year]["control"]
            bld_v   = means[year]["buildout"]
            ctrl_label = (f"${ctrl_v/1000:.0f}k" if ctrl_v >= 1000
                          else f"${ctrl_v:,.0f}")
            bld_label  = (f"${bld_v/1000:.0f}k"  if bld_v  >= 1000
                          else f"${bld_v:,.0f}")
            gap_label  = (f"${gaps_abs[i]/1000:.0f}k"
                          if abs(gaps_abs[i]) >= 1000
                          else f"${gaps_abs[i]:,.0f}")
            print(f"  {year:<10} {ctrl_label:>12} {bld_label:>12} "
                  f"{gap_label:>12} {gaps_pct[i]:>9.1f}% {net_str:>14}")

        print(f"\n  Interpretation: baseline gap = {baseline_pct:.1f}%. "
              f"By 2023 gap = {gaps_pct[2]:.1f}% "
              f"({gaps_pct[2]-baseline_pct:+.1f}pp net of baseline).")

        # ── Bar chart of gap % by period ──
        x      = np.arange(len(YEARS))
        colors = []
        for i, pct in enumerate(gaps_pct):
            net = pct - baseline_pct
            if i == 0:
                colors.append("#9CA3AF")   # baseline — neutral
            elif net > 0:
                colors.append(COLORS["buildout"])
            else:
                colors.append(COLORS["negative"])

        bars = ax.bar(x, gaps_pct, width=0.5,
                      color=colors, alpha=0.88, zorder=3)

        # Baseline reference line
        ax.axhline(baseline_pct, color="#374151", linewidth=1.5,
                   linestyle="--", alpha=0.7,
                   label=f"Baseline gap ({baseline_pct:.1f}%)")

        # Value labels
        for bar, pct, i in zip(bars, gaps_pct, range(len(YEARS))):
            net    = pct - baseline_pct if i > 0 else None
            label  = f"{pct:.1f}%"
            if net is not None:
                label += f"\n({net:+.1f}pp)"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + baseline_pct * 0.02,
                    label, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#1F2937")

        ax.set_xticks(x)
        ax.set_xticklabels(YEAR_LABELS)
        ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
        ax.legend(frameon=False, fontsize=9)
        style_ax(ax, title,
                 ylabel="Gap as % of control mean")

    fig.suptitle(
        "Buildout–Control Gap as % of Control Mean — Baseline-Controlled",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.text(
        0.5, -0.04,
        "Dashed line = 2000 baseline gap. Bar labels show gap % and "
        "change in percentage points (pp) relative to baseline.\n"
        "Positive pp = gap widened beyond pre-existing difference. "
        "Matched PSM sample (8km buffer).",
        ha="center", fontsize=8, color="#6B7280", style="italic"
    )
    plt.tight_layout()
    save_fig(fig, "08_gap_analysis.png")

    print(f"{'='*70}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    panel = pd.read_csv(PANEL_CSV, dtype={"geoid": str})
    print(f"  Panel: {len(panel)} rows | {panel['geoid'].nunique()} tracts")

    sensitivity = pd.DataFrame()
    sens_path   = "data/sensitivity_results.csv"
    if os.path.exists(sens_path):
        sensitivity = pd.read_csv(sens_path)
        print(f"  Sensitivity: {len(sensitivity)} rows loaded")
    else:
        print("  WARNING: sensitivity_results.csv not found — "
              "run Script 5 first. Skipping Chart 4.")

    print("\nGenerating all charts...")
    chart_mean_by_group(panel)
    chart_effect_size(panel)
    if not sensitivity.empty:
        chart_buffer_sensitivity(sensitivity)
    chart_absolute_change(panel)
    chart_county_comparison(panel)
    chart_gap_analysis(panel)

    print(f"\nAll charts saved to: {OUTPUT_DIR}/")
    print("Charts produced:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()