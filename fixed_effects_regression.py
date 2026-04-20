"""
Script 5 — Fixed Effects Regression (Multi-County, Three Periods)
==================================================================
Runs DiD and TWFE models on the matched panel from Script 4,
plus a buffer sensitivity analysis at 2km, 4km, 6km, and 8km.

Two treatment groups:
  control:  no DC nearby in any period (reference group)
  buildout: DC arrived anywhere in the study window 2000–2023

Model: Y_it = α_i + λ_t
            + β1(buildout × post2015)
            + β2(buildout × post2023)
            + ε_it

  α_i  = tract fixed effect (stable neighborhood characteristics)
  λ_t  = year fixed effect (statewide macro trends)

Buffer sensitivity analysis reruns PSM and TWFE at each buffer
using dist_nearest_dc_2023_km from Script 3 — no geometry
reprocessing needed.

Run after Script 4 has saved data/matched_panel.csv.

Install: pip install pandas linearmodels statsmodels
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

YEARS          = [2000, 2015, 2023]
BUFFERS_KM     = [2, 4, 6, 8]
DEFAULT_BUFFER = 8
CALIPER        = 0.10

STUDY_COUNTIES = [
    "Sacramento County",
    "Los Angeles County",
    "Alameda County",
    "San Francisco County",
]

PSM_COVARIATES = [
    "med_hh_income",
    "med_home_value",
    "med_rent",
]


# ── LOAD MATCHED PANEL ────────────────────────────────────────────────────────
# Reads the matched panel from Script 4. Contains dist_nearest_dc_2023_km
# which allows group reassignment at any buffer without touching geometry.

def load_matched_panel() -> pd.DataFrame:
    panel = pd.read_csv("data/matched_panel.csv", dtype={"geoid": str})

    # Ensure period and interaction terms exist
    panel["post_2015"] = (panel["year"] == 2015).astype(int)
    panel["post_2023"] = (panel["year"] == 2023).astype(int)

    if "is_buildout" in panel.columns:
        panel["buildout_x_post_2015"] = panel["is_buildout"] * panel["post_2015"]
        panel["buildout_x_post_2023"] = panel["is_buildout"] * panel["post_2023"]

    # Log transforms
    for col in ["med_home_value", "med_rent", "med_hh_income"]:
        if col in panel.columns:
            panel[f"log_{col}"] = np.log(panel[col].clip(lower=1))

    # Inverse distance for distance gradient model
    if "dist_nearest_dc_2023_km" in panel.columns:
        panel["inv_dist_2023"] = np.where(
            panel["dist_nearest_dc_2023_km"] > 0,
            1 / panel["dist_nearest_dc_2023_km"], np.nan
        )

    print(f"Loaded matched panel: {len(panel)} rows | "
          f"{panel['geoid'].nunique()} tracts")
    if "group" in panel.columns:
        print("\nTract counts by county and group (2000 baseline):")
        print(panel[panel["year"] == 2000].groupby(
            ["county_name", "group"])["geoid"].nunique().to_string())
    return panel


# ── REASSIGN GROUPS AT ANY BUFFER ────────────────────────────────────────────
# Uses dist_nearest_dc_2023_km to reassign treatment groups and rerun PSM
# at any buffer size. Called by the sensitivity analysis loop in main().

def reassign_and_match(panel: pd.DataFrame,
                       buffer_km: int) -> pd.DataFrame:
    panel = panel.copy()

    # Reassign groups from distance column
    panel["group"]       = np.where(
        panel["dist_nearest_dc_2023_km"] <= buffer_km,
        "buildout", "control"
    )
    panel["is_buildout"] = (panel["group"] == "buildout").astype(int)
    panel["buildout_x_post_2015"] = panel["is_buildout"] * panel["post_2015"]
    panel["buildout_x_post_2023"] = panel["is_buildout"] * panel["post_2023"]

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

        valid          = distances.flatten() <= CALIPER
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


# ── POOLED TWFE — ALL COUNTIES ────────────────────────────────────────────────
# Two-way fixed effects (tract + year).
# County dummies deliberately excluded — tract FE nests within county.

def run_pooled_twfe(panel: pd.DataFrame, outcome: str,
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
        print(f"  Pooled TWFE failed: {e}")
        return {}


# ── COUNTY-BY-COUNTY TWFE ─────────────────────────────────────────────────────
# Runs TWFE within each county separately for transparency.
# Results reported but not used as primary findings due to small samples.

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

        present_groups = sub[sub["year"] == 2000]["group"].unique()
        terms = [
            t for t in ["buildout_x_post_2015", "buildout_x_post_2023"]
            if t in sub.columns and "buildout" in present_groups
        ]
        if not terms:
            print(f"  {county:<25} {'no valid interaction terms':>48}")
            continue

        sub_idx = sub.set_index(["geoid", "year"])
        try:
            result = PanelOLS.from_formula(
                f"{outcome} ~ {' + '.join(terms)} + EntityEffects + TimeEffects",
                data=sub_idx
            ).fit(cov_type="clustered", cluster_entity=True)

            b15 = result.params.get("buildout_x_post_2015", np.nan)
            b23 = result.params.get("buildout_x_post_2023", np.nan)
            p23 = result.pvalues.get("buildout_x_post_2023", np.nan)
            sig = "***" if p23<0.01 else "**" if p23<0.05 else "*" if p23<0.1 else "n.s."
            print(f"  {county:<25} {b15:>+16,.2f} {b23:>+16,.2f} "
                  f"{p23:>10.3f} {sig:>5}")
        except Exception as e:
            print(f"  {county:<25} {'failed: '+str(e)[:40]:>48}")

    print("\n  Note: individual county models are underpowered due to small")
    print("  matched sample sizes. Pooled TWFE is the primary specification.")
    print("  County-by-county results are reported for transparency only.")


# ── SIMPLE DiD BASELINE ───────────────────────────────────────────────────────
# OLS DiD with county fixed effects and clustered SEs at the tract level.

def run_did(panel: pd.DataFrame, outcome: str) -> dict:
    if outcome not in panel.columns:
        return {}

    sub = panel.dropna(subset=[outcome]).copy()

    present_counties = sub["county_name"].unique()
    county_dummies = []
    for county in STUDY_COUNTIES:
        if county == "Sacramento County" or county not in present_counties:
            continue
        safe = county.lower().replace(" ", "_")
        sub[f"county_{safe}"] = (sub["county_name"] == county).astype(int)
        county_dummies.append(f"county_{safe}")

    county_str = " + " + " + ".join(county_dummies) if county_dummies else ""
    formula = (
        f"{outcome} ~ is_buildout + post_2015 + post_2023 "
        f"+ buildout_x_post_2015 + buildout_x_post_2023"
        f"{county_str}"
    )

    try:
        model = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["geoid"]}
        )
        print(f"\n{'─'*65}")
        print(f"DiD (OLS, clustered SE at tract level) → {outcome}")
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


# ── DISTANCE GRADIENT ─────────────────────────────────────────────────────────
# Cross-sectional OLS on 2023 data using inverse distance as continuous
# treatment. Time-invariant so cannot use tract fixed effects.

def run_distance_gradient(panel: pd.DataFrame, outcome: str):
    if outcome not in panel.columns or "inv_dist_2023" not in panel.columns:
        return

    sub = panel[
        (panel["year"] == 2023)
    ].dropna(subset=[outcome, "inv_dist_2023"]).copy()

    controls = "log_med_hh_income" if "log_med_hh_income" in sub.columns else ""
    rhs      = f"inv_dist_2023 + {controls}" if controls else "inv_dist_2023"

    present_counties = sub["county_name"].unique()
    for county in STUDY_COUNTIES:
        if county == "Sacramento County" or county not in present_counties:
            continue
        safe = county.lower().replace(" ", "_")
        sub[f"county_{safe}"] = (sub["county_name"] == county).astype(int)
        rhs += f" + county_{safe}"

    try:
        model     = smf.ols(f"{outcome} ~ {rhs}", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["geoid"]}
        )
        coef      = model.params["inv_dist_2023"]
        se        = model.bse["inv_dist_2023"]
        p         = model.pvalues["inv_dist_2023"]
        sig       = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "n.s."
        direction = "higher" if coef > 0 else "lower"
        print(f"\nDistance gradient ({outcome}, 2023 cross-section):")
        print(f"  β(inv_dist): {coef:>+,.2f}  SE={se:.2f}  p={p:.3f}  {sig}")
        print(f"  Each 1km closer to a DC → {direction} {outcome} "
              f"by {abs(coef):.2f} units")
    except Exception as e:
        print(f"  Distance gradient failed: {e}")


# ── BUFFER SENSITIVITY ANALYSIS ───────────────────────────────────────────────
# Reruns PSM + TWFE at 2km, 4km, 6km, and 8km using dist_nearest_dc_2023_km.
# No geometry reprocessing — groups reassigned on the fly from distance column.
# Produces a single comparison table showing how coefficients change with buffer.
# If findings are robust, direction and significance should be consistent
# across all four buffer sizes.

def run_buffer_sensitivity(panel: pd.DataFrame,
                           outcomes: list = None):
    if outcomes is None:
        outcomes = ["med_home_value", "med_rent"]

    print(f"\n{'='*80}")
    print("BUFFER SENSITIVITY ANALYSIS")
    print("Reruns PSM + TWFE at each buffer size using distance column from Script 3")
    print(f"{'='*80}")

    # Store results for comparison table
    sensitivity_results = []

    for buffer_km in BUFFERS_KM:
        print(f"\n── Buffer: {buffer_km}km ──────────────────────────────────────")

        matched = reassign_and_match(panel, buffer_km)
        if matched.empty:
            print(f"  No matched sample produced at {buffer_km}km — skipping")
            continue

        n_buildout = matched[
            (matched["year"] == 2000) &
            (matched["group"] == "buildout")
        ]["geoid"].nunique()
        n_control = matched[
            (matched["year"] == 2000) &
            (matched["group"] == "control")
        ]["geoid"].nunique()
        print(f"  Matched sample: {n_buildout} buildout / "
              f"{n_control} control tracts")

        for outcome in outcomes:
            r = run_pooled_twfe(
                matched, outcome,
                label=f"{buffer_km}km buffer"
            )
            for term, vals in r.items():
                sensitivity_results.append({
                    "buffer_km": buffer_km,
                    "outcome":   outcome,
                    "term":      term,
                    **vals,
                    "n_buildout": n_buildout,
                    "n_control":  n_control,
                })

    # Print comparison table
    if not sensitivity_results:
        print("\nNo sensitivity results produced.")
        return

    print(f"\n{'='*80}")
    print("SENSITIVITY COMPARISON TABLE — β buildout×post2023 by buffer size")
    print(f"{'='*80}")

    for outcome in outcomes:
        print(f"\n  {outcome}:")
        print(f"  {'Buffer':<10} {'β (2023)':>12} {'SE':>10} {'p':>8} "
              f"{'Sig':>5} {'Buildout n':>12} {'Control n':>12}")
        print("  " + "─"*65)

        sub = [r for r in sensitivity_results
               if r["outcome"] == outcome
               and r["term"] == "buildout_x_post_2023"]

        for r in sub:
            sig = ("***" if r["p"]<0.01 else "**" if r["p"]<0.05
                   else "*" if r["p"]<0.1 else "n.s.")
            print(f"  {r['buffer_km']}km{'':<6} "
                  f"{r['coef']:>+12,.2f} "
                  f"{r['se']:>10.2f} "
                  f"{r['p']:>8.3f} "
                  f"{sig:>5} "
                  f"{r['n_buildout']:>12} "
                  f"{r['n_control']:>12}")

        # Directional consistency check
        coefs = [r["coef"] for r in sub]
        if len(coefs) > 1:
            all_positive = all(c > 0 for c in coefs)
            all_negative = all(c < 0 for c in coefs)
            if all_positive:
                print(f"\n  ✓ Consistent positive direction across all buffer sizes")
            elif all_negative:
                print(f"\n  ✓ Consistent negative direction across all buffer sizes")
            else:
                print(f"\n  ! Direction is not consistent across buffer sizes — "
                      f"interpret with caution")

    # Save sensitivity results to CSV
    pd.DataFrame(sensitivity_results).to_csv(
        "data/sensitivity_results.csv", index=False
    )
    print(f"\nSensitivity results saved: data/sensitivity_results.csv")


# ── RESULTS TABLE ─────────────────────────────────────────────────────────────
# Clean summary table for the paper's results section (primary 8km spec only).

def format_results_table(results: list):
    print(f"\n{'='*75}")
    print("REGRESSION RESULTS SUMMARY (primary specification: 8km buffer)")
    print(f"{'='*75}")
    print(f"{'Outcome':<22} {'Model':<6} {'Term':<28} "
          f"{'β':>10} {'SE':>8} {'p':>7} {'Sig':>5}")
    print("─" * 75)
    for r in results:
        if not r:
            continue
        coef = r.get("coef", np.nan)
        sig  = ("***" if r["p"]<0.01 else "**" if r["p"]<0.05
                else "*" if r["p"]<0.1 else "n.s.")
        print(f"{r.get('outcome',''):<22} {r.get('model',''):<6} "
              f"{r.get('term',''):<28} {coef:>+10,.2f} "
              f"{r['se']:>8.2f} {r['p']:>7.3f} {sig:>5}")
    print("─" * 75)
    print("Clustered SEs at tract level.  * p<0.1  ** p<0.05  *** p<0.01")
    print("TWFE includes tract and year fixed effects.")
    print("DiD includes county fixed effects.")
    print("2000 = reference period.  Control = reference group.")
    print("See data/sensitivity_results.csv for full buffer sensitivity results.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    panel   = load_matched_panel()
    results = []

    outcomes = ["med_home_value", "med_rent",
                "log_med_home_value", "log_med_rent"]

    print("\n[POOLED DiD — ALL COUNTIES, 8km PRIMARY SPEC]")
    for outcome in outcomes:
        r = run_did(panel, outcome)
        for term, vals in r.items():
            results.append({**vals, "outcome": outcome,
                             "model": "DiD", "term": term})

    print("\n[POOLED TWFE — ALL COUNTIES, 8km PRIMARY SPEC]")
    for outcome in outcomes:
        r = run_pooled_twfe(panel, outcome, label="8km primary")
        for term, vals in r.items():
            results.append({**vals, "outcome": outcome,
                             "model": "TWFE", "term": term})

    print("\n[COUNTY-BY-COUNTY TWFE — GENERALIZABILITY]")
    for outcome in ["med_home_value", "med_rent"]:
        run_county_twfe(panel, outcome)

    print("\n[DISTANCE GRADIENT — CONTINUOUS TREATMENT]")
    for outcome in ["med_home_value", "med_rent"]:
        run_distance_gradient(panel, outcome)

    print("\n[BUFFER SENSITIVITY ANALYSIS — 2km / 4km / 6km / 8km]")
    run_buffer_sensitivity(panel, outcomes=["med_home_value", "med_rent"])

    format_results_table(results)


if __name__ == "__main__":
    main()