"""
Script 5 — Fixed Effects Regression (Multi-County, Three Periods)
==================================================================
Runs DiD and TWFE models on the matched panel from Script 4.

Three treatment groups:
  control:  no DC nearby in any period (reference group)
  pre2000:  DC present before 2000 (Napa Kaiser Data Center)
  buildout: DC arrived 2000–2023 (all other study facilities)

Model: Y_it = α_i + λ_t + δ_c
            + β1(pre2000 × post2015)  + β2(pre2000 × post2023)
            + β3(buildout × post2015) + β4(buildout × post2023)
            + ε_it

  α_i  = tract fixed effect (stable neighborhood characteristics)
  λ_t  = year fixed effect (statewide macro trends)
  δ_c  = county fixed effect (cross-county housing market differences)

  β3 = buildout DC effect relative to control in 2015
  β4 = buildout DC effect relative to control in 2023
  β3 → β4 tells you: did the effect grow or stabilize over time?

Run after Script 4 has saved data/matched_panel.csv.

Install: pip install pandas linearmodels statsmodels
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")

YEARS  = [2000, 2015, 2023]
GROUPS = ["pre2000", "buildout"]   # no hyperscale in this dataset

STUDY_COUNTIES = [
    "Sacramento County",
    "Los Angeles County",
    "Alameda County",
    "San Francisco County",
    # Napa excluded — no viable control tracts at any buffer size
    # Discussed descriptively as a contrast case in the paper
]


# ── LOAD MATCHED PANEL ────────────────────────────────────────────────────────
# Reads the matched panel from Script 4 and adds county dummy variables
# for the county fixed effects in the pooled regression.

def load_matched_panel() -> pd.DataFrame:
    panel = pd.read_csv("data/matched_panel.csv", dtype={"geoid": str})

    # County dummies — Sacramento is reference category
    for county in STUDY_COUNTIES:
        if county == "Sacramento County":
            continue
        safe = county.lower().replace(" ", "_")
        panel[f"county_{safe}"] = (panel["county_name"] == county).astype(int)

    # Ensure period and interaction terms exist
    panel["post_2015"] = (panel["year"] == 2015).astype(int)
    panel["post_2023"] = (panel["year"] == 2023).astype(int)
    for group in GROUPS:
        for post in ["post_2015", "post_2023"]:
            col = f"{group}_x_{post}"
            if col not in panel.columns and f"is_{group}" in panel.columns:
                panel[col] = panel[f"is_{group}"] * panel[post]

    print(f"Loaded matched panel: {len(panel)} rows | "
          f"{panel['geoid'].nunique()} tracts")
    if "group" in panel.columns:
        print(panel[panel["year"] == 2000].groupby(
            ["county_name", "group"])["geoid"].nunique().to_string())
    return panel


# ── POOLED TWFE — ALL COUNTIES ────────────────────────────────────────────────
# Two-way fixed effects with county fixed effects.
# β3 and β4 are the main effects of interest (buildout group).
# β1 and β2 capture the pre2000 long-exposure effect (Napa contrast case).

def run_pooled_twfe(panel: pd.DataFrame, outcome: str) -> dict:
    if outcome not in panel.columns:
        print(f"  Skipping {outcome} — column not found")
        return {}

    sub = panel.dropna(subset=[outcome]).copy()

    # All group × period interactions plus county dummies
    interaction_terms = [
        f"{g}_x_{p}"
        for g in GROUPS
        for p in ["post_2015", "post_2023"]
        if f"{g}_x_{p}" in sub.columns
    ]
    county_dummies = [c for c in sub.columns if c.startswith("county_")]
    rhs = " + ".join(interaction_terms + county_dummies)

    sub_idx = sub.set_index(["geoid", "year"])
    try:
        model  = PanelOLS.from_formula(
            f"{outcome} ~ {rhs} + EntityEffects + TimeEffects",
            data=sub_idx
        )
        result = model.fit(cov_type="clustered", cluster_entity=True)

        print(f"\n{'─'*65}")
        print(f"POOLED TWFE (all counties, 3 periods) → {outcome}")
        print(f"  R² within: {result.rsquared_within:.3f}  N={result.nobs}")
        print(f"  {'Term':<28} {'β':>10} {'SE':>8} {'p':>7} {'Sig':>5}")
        print(f"  {'─'*55}")

        labels = {
            "pre2000_x_post_2015":  "Pre-2000 DC × 2015  ",
            "pre2000_x_post_2023":  "Pre-2000 DC × 2023  ",
            "buildout_x_post_2015": "Buildout DC × 2015  ",
            "buildout_x_post_2023": "Buildout DC × 2023  ",
        }
        out = {}
        for term, label in labels.items():
            if term not in result.params:
                continue
            coef = result.params[term]
            se   = result.std_errors[term]
            p    = result.pvalues[term]
            ci   = result.conf_int().loc[term]
            sig  = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "n.s."
            print(f"  {label:<28} {coef:>+10,.2f} {se:>8.2f} {p:>7.3f} {sig:>5}")
            out[term] = {"coef": coef, "se": se, "p": p,
                         "ci_lo": ci["lower"], "ci_hi": ci["upper"]}

        # Did the buildout effect grow or shrink from 2015 to 2023?
        t15 = "buildout_x_post_2015"
        t23 = "buildout_x_post_2023"
        if t15 in out and t23 in out:
            grew = out[t23]["coef"] > out[t15]["coef"]
            print(f"\n  Buildout effect {'grew' if grew else 'shrank'} "
                  f"2015→2023  "
                  f"({out[t15]['coef']:+,.0f} → {out[t23]['coef']:+,.0f})")
        return out

    except Exception as e:
        print(f"  Pooled TWFE failed: {e}")
        return {}


# ── COUNTY-BY-COUNTY TWFE ─────────────────────────────────────────────────────
# Runs the same model within each county separately.
# Consistent direction across counties = finding generalizes.
# Napa only has pre2000 tracts so buildout results will be empty there.

def run_county_twfe(panel: pd.DataFrame, outcome: str):
    if outcome not in panel.columns:
        return

    print(f"\n{'─'*70}")
    print(f"COUNTY-BY-COUNTY TWFE → {outcome}")
    print(f"{'County':<25} {'β buildout×2015':>16} {'β buildout×2023':>16} "
          f"{'p (2023)':>10} {'Sig':>5}")
    print("─" * 70)

    for county in STUDY_COUNTIES:
        sub = panel[
            (panel["county_name"] == county) &
            panel[outcome].notna()
        ].copy()

        if "is_buildout" not in sub.columns or sub["is_buildout"].sum() == 0:
            print(f"  {county:<25} {'no buildout tracts':>48}")
            continue

        sub_idx = sub.set_index(["geoid", "year"])
        try:
            terms = " + ".join([
                f"{g}_x_{p}"
                for g in GROUPS
                for p in ["post_2015", "post_2023"]
                if f"{g}_x_{p}" in sub.columns
            ])
            model  = PanelOLS.from_formula(
                f"{outcome} ~ {terms} + EntityEffects + TimeEffects",
                data=sub_idx
            )
            result = model.fit(cov_type="clustered", cluster_entity=True)

            b15 = result.params.get("buildout_x_post_2015", np.nan)
            b23 = result.params.get("buildout_x_post_2023", np.nan)
            p23 = result.pvalues.get("buildout_x_post_2023", np.nan)
            sig = "***" if p23<0.01 else "**" if p23<0.05 else "*" if p23<0.1 else "n.s."
            print(f"  {county:<25} {b15:>+16,.2f} {b23:>+16,.2f} "
                  f"{p23:>10.3f} {sig:>5}")
        except Exception as e:
            print(f"  {county:<25} {'failed: '+str(e)[:35]:>45}")

    print("\n  Consistent direction across counties → finding generalizes")
    print("  Napa skipped — only pre2000 tracts (contrast case, not buildout)")


# ── SIMPLE DiD BASELINE ───────────────────────────────────────────────────────
# OLS DiD without fixed effects as a baseline comparison.
# If DiD and TWFE coefficients are similar, fixed effects aren't
# doing unexpected work — builds confidence in the TWFE result.

def run_did(panel: pd.DataFrame, outcome: str) -> dict:
    if outcome not in panel.columns:
        return {}

    sub = panel.dropna(subset=[outcome]).copy()
    county_dummies = " + ".join(
        [c for c in sub.columns if c.startswith("county_")]
    )
    interaction_terms = " + ".join([
        f"{g}_x_{p}" for g in GROUPS
        for p in ["post_2015", "post_2023"]
        if f"{g}_x_{p}" in sub.columns
    ])
    group_terms = " + ".join([f"is_{g}" for g in GROUPS
                               if f"is_{g}" in sub.columns])
    formula = (f"{outcome} ~ {group_terms} + post_2015 + post_2023 "
               f"+ {interaction_terms}"
               + (f" + {county_dummies}" if county_dummies else ""))

    model = smf.ols(formula, data=sub).fit(cov_type="HC3")

    print(f"\n{'─'*65}")
    print(f"DiD (OLS with county controls) → {outcome}")
    out = {}
    for term in [f"{g}_x_{p}" for g in GROUPS
                 for p in ["post_2015", "post_2023"]]:
        if term not in model.params:
            continue
        coef = model.params[term]
        se   = model.bse[term]
        p    = model.pvalues[term]
        sig  = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "n.s."
        print(f"  {term:<30} β={coef:+,.2f}  SE={se:.2f}  p={p:.3f}  {sig}")
        out[term] = {"coef": coef, "se": se, "p": p}
    return out


# ── DISTANCE GRADIENT ─────────────────────────────────────────────────────────
# Continuous treatment: inverse distance to nearest DC in 2023.
# If significant, supports a dose-response narrative in your paper.

def run_distance_gradient(panel: pd.DataFrame, outcome: str):
    if outcome not in panel.columns or "inv_dist_2023" not in panel.columns:
        return

    sub     = panel.dropna(subset=[outcome, "inv_dist_2023"]).copy()
    sub_idx = sub.set_index(["geoid", "year"])
    controls = "log_med_hh_income" if "log_med_hh_income" in sub.columns else ""
    rhs      = f"inv_dist_2023 + {controls}" if controls else "inv_dist_2023"

    try:
        result = PanelOLS.from_formula(
            f"{outcome} ~ {rhs} + EntityEffects + TimeEffects",
            data=sub_idx
        ).fit(cov_type="clustered", cluster_entity=True)

        coef      = result.params["inv_dist_2023"]
        p         = result.pvalues["inv_dist_2023"]
        sig       = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "n.s."
        direction = "higher" if coef > 0 else "lower"
        print(f"\nDistance gradient ({outcome}):")
        print(f"  β(inv_dist): {coef:+,.2f}  p={p:.3f}  {sig}")
        print(f"  Each 1km closer to a DC → {direction} {outcome} "
              f"by {abs(coef):.2f} units")
    except Exception as e:
        print(f"  Distance gradient failed: {e}")


# ── RESULTS TABLE ─────────────────────────────────────────────────────────────
# Clean summary table for your paper's results section.

def format_results_table(results: list):
    print(f"\n{'='*70}")
    print("REGRESSION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Outcome':<22} {'Model':<6} {'Term':<28} "
          f"{'β':>10} {'SE':>8} {'p':>7} {'Sig':>5}")
    print("─" * 70)
    for r in results:
        if not r:
            continue
        coef = r.get("coef", np.nan)
        sig  = ("***" if r["p"]<0.01 else "**" if r["p"]<0.05
                else "*" if r["p"]<0.1 else "n.s.")
        print(f"{r.get('outcome',''):<22} {r.get('model',''):<6} "
              f"{r.get('term',''):<28} {coef:>+10,.2f} "
              f"{r['se']:>8.2f} {r['p']:>7.3f} {sig:>5}")
    print("─" * 70)
    print("Clustered SEs at tract level.  * p<0.1  ** p<0.05  *** p<0.01")
    print("TWFE includes tract, year, and county fixed effects.")
    print("2000 = reference period.  Control = reference group.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    panel   = load_matched_panel()
    results = []

    outcomes = ["med_home_value", "med_rent",
                "log_med_home_value", "log_med_rent"]

    print("\n[POOLED DiD — ALL COUNTIES]")
    for outcome in outcomes:
        r = run_did(panel, outcome)
        for term, vals in r.items():
            results.append({**vals, "outcome": outcome,
                             "model": "DiD", "term": term})

    print("\n[POOLED TWFE — ALL COUNTIES]")
    for outcome in outcomes:
        r = run_pooled_twfe(panel, outcome)
        for term, vals in r.items():
            results.append({**vals, "outcome": outcome,
                             "model": "TWFE", "term": term})

    print("\n[COUNTY-BY-COUNTY TWFE — GENERALIZABILITY]")
    for outcome in ["med_home_value", "med_rent"]:
        run_county_twfe(panel, outcome)

    print("\n[DISTANCE GRADIENT — CONTINUOUS TREATMENT]")
    for outcome in ["med_home_value", "med_rent"]:
        run_distance_gradient(panel, outcome)

    format_results_table(results)


if __name__ == "__main__":
    main()