"""
Script 5b — Fixed Effects Regression (FHFA Robustness Check)
=============================================================
Identical to Script 5 but reads from data/matched_panel_fhfa.csv
and produces a side-by-side comparison table with the primary
Census results. The primary matched_panel.csv is never touched.

The key question: do the TWFE coefficients from FHFA transaction
data match the Census ACS estimates in direction and magnitude?
If yes, the finding is robust to data source.

Run after Script 4b — bias_reduction_fhfa.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")

# ── FHFA-specific paths ────────────────────────────────────────────────────────
FHFA_CSV   = "data/matched_panel_fhfa.csv"
CENSUS_CSV = "data/matched_panel.csv"

YEARS  = [2000, 2015, 2023]

STUDY_COUNTIES = [
    "Sacramento County",
    "Los Angeles County",
    "Alameda County",
    "San Francisco County",
]

# Primary Census TWFE results — hardcoded for comparison table
# Update if you rerun Script 5 with different settings
CENSUS_RESULTS = {
    "med_home_value": {
        "buildout_x_post_2015": {"coef": -1171.45,   "se": 26366.71, "p": 0.965},
        "buildout_x_post_2023": {"coef": 123306.97,  "se": 46097.27, "p": 0.008},
    },
    "med_rent": {
        "buildout_x_post_2015": {"coef": 48.19,  "se": 43.44, "p": 0.268},
        "buildout_x_post_2023": {"coef": 133.90, "se": 67.16, "p": 0.047},
    },
}


def load_matched_panel(csv_path: str, label: str) -> pd.DataFrame:
    panel = pd.read_csv(csv_path, dtype={"geoid": str})

    panel["post_2015"] = (panel["year"] == 2015).astype(int)
    panel["post_2023"] = (panel["year"] == 2023).astype(int)

    if "is_buildout" in panel.columns:
        panel["buildout_x_post_2015"] = panel["is_buildout"] * panel["post_2015"]
        panel["buildout_x_post_2023"] = panel["is_buildout"] * panel["post_2023"]

    for col in ["med_home_value", "med_rent", "med_hh_income"]:
        if col in panel.columns:
            panel[f"log_{col}"] = np.log(panel[col].clip(lower=1))

    if "dist_nearest_dc_2023_km" in panel.columns:
        panel["inv_dist_2023"] = np.where(
            panel["dist_nearest_dc_2023_km"] > 0,
            1 / panel["dist_nearest_dc_2023_km"], np.nan
        )

    print(f"\n{label} panel: {len(panel)} rows | "
          f"{panel['geoid'].nunique()} tracts")
    if "group" in panel.columns:
        print(panel[panel["year"] == 2000].groupby(
            ["county_name", "group"])["geoid"].nunique().to_string())
    return panel


def run_pooled_twfe(panel: pd.DataFrame,
                    outcome: str,
                    label: str = "") -> dict:
    if outcome not in panel.columns:
        return {}

    sub = panel.dropna(subset=[outcome]).copy()
    interaction_terms = [
        t for t in ["buildout_x_post_2015", "buildout_x_post_2023"]
        if t in sub.columns
    ]
    if not interaction_terms:
        return {}

    sub_idx = sub.set_index(["geoid", "year"])
    try:
        model  = PanelOLS.from_formula(
            f"{outcome} ~ {' + '.join(interaction_terms)} "
            f"+ EntityEffects + TimeEffects",
            data=sub_idx
        )
        result = model.fit(cov_type="clustered", cluster_entity=True)

        header = f"POOLED TWFE → {outcome}"
        if label:
            header += f"  [{label}]"
        print(f"\n{'─'*65}")
        print(header)
        print(f"  R² within: {result.rsquared_within:.3f}  N={result.nobs}")
        print(f"  {'Term':<28} {'β':>10} {'SE':>8} {'p':>7} {'95% CI'}")
        print(f"  {'─'*65}")

        labels = {
            "buildout_x_post_2015": "Buildout DC × 2015",
            "buildout_x_post_2023": "Buildout DC × 2023",
        }
        out = {}
        for term, lbl in labels.items():
            if term not in result.params:
                continue
            coef = result.params[term]
            se   = result.std_errors[term]
            p    = result.pvalues[term]
            ci   = result.conf_int().loc[term]
            sig  = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "n.s."
            print(f"  {lbl:<28} {coef:>+10,.2f} {se:>8.2f} {p:>7.3f} {sig}  "
                  f"[{ci['lower']:+,.2f}, {ci['upper']:+,.2f}]")
            out[term] = {"coef": coef, "se": se, "p": p,
                         "ci_lo": ci["lower"], "ci_hi": ci["upper"]}

        t15, t23 = "buildout_x_post_2015", "buildout_x_post_2023"
        if t15 in out and t23 in out:
            grew = out[t23]["coef"] > out[t15]["coef"]
            print(f"\n  Effect {'grew' if grew else 'shrank'} 2015→2023  "
                  f"({out[t15]['coef']:+,.0f} → {out[t23]['coef']:+,.0f})")
        return out

    except Exception as e:
        print(f"  TWFE failed: {e}")
        return {}


def run_did(panel: pd.DataFrame, outcome: str,
            label: str = "") -> dict:
    if outcome not in panel.columns:
        return {}

    sub = panel.dropna(subset=[outcome]).copy()

    present_counties = sub["county_name"].unique()
    county_dummies   = []
    for county in STUDY_COUNTIES:
        if county == "Sacramento County" or county not in present_counties:
            continue
        safe = county.lower().replace(" ", "_")
        sub[f"county_{safe}"] = (sub["county_name"] == county).astype(int)
        county_dummies.append(f"county_{safe}")

    county_str = " + " + " + ".join(county_dummies) if county_dummies else ""
    formula    = (
        f"{outcome} ~ is_buildout + post_2015 + post_2023 "
        f"+ buildout_x_post_2015 + buildout_x_post_2023"
        f"{county_str}"
    )

    try:
        model = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["geoid"]}
        )
        header = f"DiD → {outcome}"
        if label:
            header += f"  [{label}]"
        print(f"\n{'─'*65}")
        print(header)
        out = {}
        for term in ["buildout_x_post_2015", "buildout_x_post_2023"]:
            if term not in model.params:
                continue
            coef = model.params[term]
            se   = model.bse[term]
            p    = model.pvalues[term]
            sig  = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "n.s."
            print(f"  {term:<30} β={coef:>+10,.2f}  SE={se:>8.2f}  "
                  f"p={p:.3f}  {sig}")
            out[term] = {"coef": coef, "se": se, "p": p}
        return out
    except Exception as e:
        print(f"  DiD failed: {e}")
        return {}


# ── COMPARISON TABLE ──────────────────────────────────────────────────────────
# Side-by-side comparison of Census and FHFA TWFE results.
# This is the key robustness check output — if both columns show
# significant positive coefficients the finding is robust.

def print_comparison_table(fhfa_results: dict):
    print(f"\n{'='*80}")
    print("ROBUSTNESS CHECK — CENSUS ACS vs FHFA HPI (TWFE, buildout × post2023)")
    print(f"{'='*80}")
    print(f"{'Outcome':<22} {'':^5} {'Census ACS (primary)':^28} "
          f"{'FHFA HPI (robustness)':^28}")
    print(f"{'':22} {'':5} {'β':>8} {'SE':>8} {'p':>6} {'Sig':>5}  "
          f"{'β':>8} {'SE':>8} {'p':>6} {'Sig':>5}")
    print("─" * 80)

    term = "buildout_x_post_2023"
    for outcome in ["med_home_value", "med_rent",
                    "log_med_home_value", "log_med_rent"]:
        census = CENSUS_RESULTS.get(outcome, {}).get(term, {})
        fhfa   = fhfa_results.get(outcome, {}).get(term, {})

        c_coef = census.get("coef", np.nan)
        c_se   = census.get("se",   np.nan)
        c_p    = census.get("p",    np.nan)
        c_sig  = ("***" if c_p<0.01 else "**" if c_p<0.05
                  else "*" if c_p<0.1 else "n.s."
                  if not np.isnan(c_p) else "—")

        f_coef = fhfa.get("coef", np.nan)
        f_se   = fhfa.get("se",   np.nan)
        f_p    = fhfa.get("p",    np.nan)
        f_sig  = ("***" if f_p<0.01 else "**" if f_p<0.05
                  else "*" if f_p<0.1 else "n.s."
                  if not np.isnan(f_p) else "—")

        # Direction match indicator
        if not np.isnan(c_coef) and not np.isnan(f_coef):
            direction = "✓" if (c_coef > 0) == (f_coef > 0) else "✗"
        else:
            direction = "?"

        c_coef_str = f"{c_coef:>+10,.2f}" if not np.isnan(c_coef) else f"{'—':>10}"
        f_coef_str = f"{f_coef:>+10,.2f}" if not np.isnan(f_coef) else f"{'—':>10}"
        c_se_str   = f"{c_se:>8.2f}"      if not np.isnan(c_se)   else f"{'—':>8}"
        f_se_str   = f"{f_se:>8.2f}"      if not np.isnan(f_se)   else f"{'—':>8}"
        c_p_str    = f"{c_p:>6.3f}"       if not np.isnan(c_p)    else f"{'—':>6}"
        f_p_str    = f"{f_p:>6.3f}"       if not np.isnan(f_p)    else f"{'—':>6}"

        print(f"{outcome:<22} {direction:^5} "
              f"{c_coef_str} {c_se_str} {c_p_str} {c_sig:>5}  "
              f"{f_coef_str} {f_se_str} {f_p_str} {f_sig:>5}")

    print("─" * 80)
    print("✓ = same direction   ✗ = opposite direction")
    print("Census ACS: self-reported owner estimates (Script 2)")
    print("FHFA HPI:   repeat-sales transaction data (Script 2b)")
    print("\nInterpretation:")
    print("  Consistent direction + both significant → strong robustness")
    print("  Consistent direction + FHFA n.s.        → directional support")
    print("  Opposite direction                       → warrants discussion")


# ── BUFFER SENSITIVITY (FHFA) ────────────────────────────────────────────────

def reassign_and_match(panel: pd.DataFrame,
                       buffer_km: int) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    panel = panel.copy()
    panel["group"]       = np.where(
        panel["dist_nearest_dc_2023_km"] <= buffer_km,
        "buildout", "control"
    )
    panel["is_buildout"] = (panel["group"] == "buildout").astype(int)
    panel["buildout_x_post_2015"] = panel["is_buildout"] * panel["post_2015"]
    panel["buildout_x_post_2023"] = panel["is_buildout"] * panel["post_2023"]

    PSM_COVARIATES = ["med_hh_income", "med_home_value", "med_rent"]
    matched_frames = []

    for county in STUDY_COUNTIES:
        sub = panel[
            (panel["county_name"] == county) &
            (panel["year"] == 2000)
        ].copy()

        available_covs = [c for c in PSM_COVARIATES if c in sub.columns]
        clean   = sub.dropna(subset=available_covs + ["is_buildout"])
        treated = clean[clean["is_buildout"] == 1]
        control = clean[clean["group"] == "control"]

        if len(treated) == 0 or len(control) == 0:
            continue

        scaler = StandardScaler()
        X      = scaler.fit_transform(clean[available_covs].values)
        y      = (clean["group"] != "control").astype(int).values
        lr     = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        clean  = clean.copy()
        clean["ps"] = lr.predict_proba(X)[:, 1]

        treated_ps = clean[clean["is_buildout"] == 1][["geoid", "ps"]]
        control_ps = clean[clean["group"] == "control"][["geoid", "ps"]]

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(control_ps[["ps"]].values)
        distances, indices = nn.kneighbors(treated_ps[["ps"]].values)

        valid          = distances.flatten() <= 0.10
        matched_treat  = treated_ps["geoid"].values[valid]
        matched_ctrl   = control_ps["geoid"].values[indices.flatten()[valid]]
        matched_geoids = set(matched_treat) | set(matched_ctrl)

        matched_frames.append(
            panel[(panel["county_name"] == county) &
                  (panel["geoid"].isin(matched_geoids))]
        )

    if not matched_frames:
        return pd.DataFrame()
    return pd.concat(matched_frames, ignore_index=True)


def run_buffer_sensitivity(panel: pd.DataFrame):
    BUFFERS_KM = [2, 4, 6, 8]
    outcomes   = ["med_home_value", "med_rent"]

    print(f"\n{'='*75}")
    print("FHFA BUFFER SENSITIVITY ANALYSIS")
    print(f"{'='*75}")

    sensitivity_results = []

    for buffer_km in BUFFERS_KM:
        print(f"\n── Buffer: {buffer_km}km ──")
        matched = reassign_and_match(panel, buffer_km)
        if matched.empty:
            print(f"  No matched sample at {buffer_km}km")
            continue

        n_buildout = matched[
            (matched["year"] == 2000) &
            (matched["group"] == "buildout")
        ]["geoid"].nunique()
        n_control = matched[
            (matched["year"] == 2000) &
            (matched["group"] == "control")
        ]["geoid"].nunique()
        print(f"  {n_buildout} buildout / {n_control} control tracts")

        for outcome in outcomes:
            r = run_pooled_twfe(matched, outcome,
                                label=f"FHFA {buffer_km}km")
            for term, vals in r.items():
                sensitivity_results.append({
                    "buffer_km":  buffer_km,
                    "outcome":    outcome,
                    "term":       term,
                    "n_buildout": n_buildout,
                    "n_control":  n_control,
                    **vals,
                })

    if sensitivity_results:
        print(f"\n{'='*75}")
        print("FHFA SENSITIVITY — β buildout×post2023 by buffer")
        print(f"{'='*75}")
        for outcome in outcomes:
            print(f"\n  {outcome}:")
            print(f"  {'Buffer':<10} {'β':>12} {'SE':>10} {'p':>8} "
                  f"{'Sig':>5} {'n_bld':>8} {'n_ctrl':>8}")
            print("  " + "─" * 60)
            for r in [x for x in sensitivity_results
                      if x["outcome"] == outcome
                      and x["term"] == "buildout_x_post_2023"]:
                sig = ("***" if r["p"]<0.01 else "**" if r["p"]<0.05
                       else "*" if r["p"]<0.1 else "n.s.")
                print(f"  {r['buffer_km']}km{'':<6} "
                      f"{r['coef']:>+12,.2f} {r['se']:>10.2f} "
                      f"{r['p']:>8.3f} {sig:>5} "
                      f"{r['n_buildout']:>8} {r['n_control']:>8}")

        pd.DataFrame(sensitivity_results).to_csv(
            "data/sensitivity_results_fhfa.csv", index=False
        )
        print("\nFHFA sensitivity results saved: "
              "data/sensitivity_results_fhfa.csv")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Script 5b — FHFA Robustness Check")
    print(f"Reading FHFA panel from: {FHFA_CSV}")

    panel   = load_matched_panel(FHFA_CSV, "FHFA")
    outcomes = ["med_home_value", "med_rent",
                "log_med_home_value", "log_med_rent"]

    fhfa_results = {}

    print("\n[FHFA DiD — ALL COUNTIES]")
    for outcome in outcomes:
        run_did(panel, outcome, label="FHFA")

    print("\n[FHFA TWFE — ALL COUNTIES]")
    for outcome in outcomes:
        r = run_pooled_twfe(panel, outcome, label="FHFA")
        if r:
            fhfa_results[outcome] = r

    print("\n[FHFA BUFFER SENSITIVITY]")
    run_buffer_sensitivity(panel)

    # Key output — side-by-side comparison with Census results
    print_comparison_table(fhfa_results)

    print(f"\n{'='*65}")
    print("FILES PRODUCED BY FHFA ROBUSTNESS CHECK")
    print(f"{'='*65}")
    print(f"  {FHFA_CSV}")
    print(f"  data/sensitivity_results_fhfa.csv")
    print(f"\nPrimary Census files unchanged:")
    print(f"  {CENSUS_CSV}")
    print(f"  data/sensitivity_results.csv")


if __name__ == "__main__":
    main()
