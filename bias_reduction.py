"""
Script 4 — Bias Reduction (Multi-County, Three Periods)
=========================================================
Runs propensity score matching within each study county using
2000 baseline covariates, then combines matched samples into
one panel for regression.

Matching on 2000 covariates means we are comparing tracts that
looked similar BEFORE any study DC existed — the strongest
possible baseline for a difference-in-differences design.

Outputs:
  - Balance table printed to console (Table 1 in paper)
  - data/matched_panel.csv (input for Script 5)

Run after Scripts 2 and 3 are complete.

Install: pip install pandas geopandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

GPKG_PATH = "geopackages/housing_data.gpkg"
YEARS     = [2000, 2015, 2023]

STUDY_COUNTIES = [
    "Sacramento County",
    "Los Angeles County",
    "Alameda County",
    "San Francisco County",
    # Napa excluded — no viable control tracts at any buffer size
]

# PSM covariates — generic names used after panel reshape
# (the panel has year=2000 rows with columns named med_home_value,
# med_rent, med_hh_income — not med_home_value_2000 etc.)
PSM_COVARIATES = [
    "med_hh_income",
    "med_home_value",
    "med_rent",
]

CALIPER = 0.10   # max propensity score difference for a valid match


# ── LOAD AND RESHAPE TO PANEL ─────────────────────────────────────────────────
# Reads the tract layer and reshapes from wide to long panel format.
# Creates all DiD interaction terms needed by Script 5.

def load_panel() -> pd.DataFrame:
    gdf = gpd.read_file(GPKG_PATH, layer="study_tracts")

    # Diagnostic — print all columns so we can see what actually came through
    print("\nColumns in study_tracts layer:")
    for col in sorted(gdf.columns):
        print(f"  {col}")
    print()

    rows = []
    for year in YEARS:
        cols = {
            f"med_home_value_{year}": "med_home_value",
            f"med_rent_{year}":       "med_rent",
            f"med_hh_income_{year}":  "med_hh_income",
            f"renter_ratio_{year}":   "renter_ratio",
            f"rent_burden_{year}":    "rent_burden",
        }
        available = {k: v for k, v in cols.items() if k in gdf.columns}

        # Grab treatment columns — handle case where they may not exist yet
        treatment_cols = [c for c in ["group", "is_pre2000", "is_buildout",
                                       "is_hyperscale", "near_dc_2000",
                                       "near_dc_2015", "near_dc_2023",
                                       "dist_nearest_dc_2000_km",
                                       "dist_nearest_dc_2015_km",
                                       "dist_nearest_dc_2023_km"]
                          if c in gdf.columns]

        yr_df = gdf[["geoid", "county_name"] +
                    treatment_cols +
                    list(available.keys())].copy()
        yr_df = yr_df.rename(columns=available)
        yr_df["year"] = year
        rows.append(yr_df)

    panel = pd.concat(rows, ignore_index=True)
    if "geometry" in panel.columns:
        panel = panel.drop(columns=["geometry"])

    # Period indicators — 2000 is the reference period
    panel["post_2015"] = (panel["year"] == 2015).astype(int)
    panel["post_2023"] = (panel["year"] == 2023).astype(int)

    # DiD interactions: each group × each post-period
    for group in ["pre2000", "buildout"]:
        if f"is_{group}" in panel.columns:
            for post in ["post_2015", "post_2023"]:
                panel[f"{group}_x_{post}"] = panel[f"is_{group}"] * panel[post]

    # Log transforms for regression
    for col in ["med_home_value", "med_rent", "med_hh_income"]:
        if col in panel.columns:
            panel[f"log_{col}"] = np.log(panel[col].clip(lower=1))

    # Continuous treatment: inverse distance to nearest DC in 2023
    if "dist_nearest_dc_2023_km" in panel.columns:
        panel["inv_dist_2023"] = np.where(
            panel["dist_nearest_dc_2023_km"] > 0,
            1 / panel["dist_nearest_dc_2023_km"], np.nan
        )

    print(f"Panel: {len(panel)} rows | "
          f"{panel['geoid'].nunique()} tracts | 3 periods")

    # Diagnostic — show which 2000 columns have actual data
    print("\n2000 covariate availability:")
    yr2000 = panel[panel["year"] == 2000]
    for col in PSM_COVARIATES:
        if col in yr2000.columns:
            n_valid = yr2000[col].notna().sum()
            print(f"  {col}: {n_valid} non-null values out of {len(yr2000)}")
        else:
            print(f"  {col}: column not found")

    if "group" in panel.columns:
        print(panel[panel["year"] == 2000].groupby(
            ["county_name", "group"])["geoid"].nunique().to_string())
    return panel


# ── PROPENSITY SCORE MATCHING ─────────────────────────────────────────────────
# Matches treated tracts to control tracts within each county separately.
# Uses 2000 covariates so matching reflects pre-treatment similarity.

def match_within_county(panel: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning propensity score matching within each county...")
    matched_frames = []

    for county in STUDY_COUNTIES:
        # Use only 2000 observations for matching
        sub = panel[
            (panel["county_name"] == county) &
            (panel["year"] == 2000)
        ].copy()

        available_covs = [c for c in PSM_COVARIATES if c in sub.columns]
        if len(available_covs) < 2:
            print(f"  {county}: not enough 2000 covariates — skipping")
            continue

        if "is_buildout" not in sub.columns:
            print(f"  {county}: treatment columns missing — run Script 3 first")
            continue

        clean   = sub.dropna(subset=available_covs + ["is_buildout"])
        treated = clean[clean["is_buildout"] == 1]
        control = clean[clean["group"] == "control"]

        if len(treated) == 0 or len(control) == 0:
            print(f"  {county}: no treated or control tracts — skipping")
            # Still include pre2000 tracts if they exist
            pre2000_geoids = set(
                sub[sub["group"] == "pre2000"]["geoid"].values
            )
            if pre2000_geoids:
                matched_frames.append(
                    panel[(panel["county_name"] == county) &
                          (panel["geoid"].isin(pre2000_geoids))]
                )
                print(f"    Added {len(pre2000_geoids)} pre2000 tracts from {county}")
            continue

        # Fit logistic regression to estimate propensity to receive a DC
        scaler = StandardScaler()
        X      = scaler.fit_transform(clean[available_covs].values)
        y      = (clean["group"] != "control").astype(int).values
        lr     = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        clean  = clean.copy()
        clean["ps"] = lr.predict_proba(X)[:, 1]

        treated_ps = clean[clean["is_buildout"] == 1][["geoid", "ps"]]
        control_ps = clean[clean["group"] == "control"][["geoid", "ps"]]

        # 1:1 nearest neighbor matching with caliper
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(control_ps[["ps"]].values)
        distances, indices = nn.kneighbors(treated_ps[["ps"]].values)

        valid          = distances.flatten() <= CALIPER
        matched_treat  = treated_ps["geoid"].values[valid]
        matched_ctrl   = control_ps["geoid"].values[indices.flatten()[valid]]
        matched_geoids = set(matched_treat) | set(matched_ctrl)

        # Also include all pre2000 tracts in this county
        pre2000_geoids = set(
            sub[sub["group"] == "pre2000"]["geoid"].values
        )
        all_geoids = matched_geoids | pre2000_geoids

        n_dropped = (~valid).sum()
        print(f"  {county}: {len(matched_treat)} matched buildout/control pairs "
              f"({n_dropped} outside caliper) + "
              f"{len(pre2000_geoids)} pre2000 tracts")

        # Store propensity scores back in full panel
        ps_map     = clean.set_index("geoid")["ps"].to_dict()
        county_idx = panel["county_name"] == county
        panel.loc[county_idx, "propensity_score"]  = (
            panel.loc[county_idx, "geoid"].map(ps_map)
        )
        panel.loc[county_idx, "in_matched_sample"] = (
            panel.loc[county_idx, "geoid"].isin(all_geoids)
        )

        matched_frames.append(
            panel[(panel["county_name"] == county) &
                  (panel["geoid"].isin(all_geoids))]
        )

    if not matched_frames:
        print("WARNING: No matched samples produced — check treatment groups")
        return panel

    matched = pd.concat(matched_frames, ignore_index=True)
    print(f"\nMatched panel: {matched['geoid'].nunique()} tracts "
          f"across {matched['county_name'].nunique()} counties")
    print(matched[matched["year"]==2000].groupby(
        ["county_name", "group"])["geoid"].nunique().to_string())
    return matched


# ── BALANCE TABLE ─────────────────────────────────────────────────────────────
# Compares treated vs control on 2000 baseline covariates before and after
# matching. SMD < 0.1 indicates good balance. Use as Table 1 in paper.

def balance_table(full_panel: pd.DataFrame, matched_panel: pd.DataFrame):
    covariates = [c for c in PSM_COVARIATES if c in full_panel.columns]

    if not covariates:
        print("WARNING: No 2000 covariates found for balance table")
        return

    print(f"\n{'='*75}")
    print("BALANCE TABLE — 2000 baseline covariates")
    print(f"{'='*75}")
    print(f"{'Variable':<28} {'Before matching':^28} {'After matching':^28}")
    print(f"{'':28} {'Treat':>8} {'Control':>8} {'SMD':>8}  "
          f"{'Treat':>8} {'Control':>8} {'SMD':>8}")
    print("─" * 75)

    pre_full    = full_panel[full_panel["year"] == 2000]
    pre_matched = matched_panel[matched_panel["year"] == 2000]

    for cov in covariates:
        t_bf  = pre_full[pre_full["is_buildout"] == 1][cov].dropna()
        c_bf  = pre_full[pre_full["group"] == "control"][cov].dropna()
        smd_b = _smd(t_bf, c_bf)

        t_af  = pre_matched[pre_matched["is_buildout"] == 1][cov].dropna()
        c_af  = pre_matched[pre_matched["group"] == "control"][cov].dropna()
        smd_a = _smd(t_af, c_af)
        flag  = "✓" if abs(smd_a) < 0.1 else "!"

        print(f"{cov:<28} {t_bf.mean():>8,.0f} {c_bf.mean():>8,.0f} "
              f"{smd_b:>8.3f}  {t_af.mean():>8,.0f} {c_af.mean():>8,.0f} "
              f"{smd_a:>8.3f} {flag}")

    print("─" * 75)
    print("SMD = standardized mean difference  ✓ < 0.1  ! >= 0.1")
    print("Note: pre2000 tracts included but not PSM-matched —")
    print("      they enter the regression with tract fixed effects only.")
    print("\nBy county (matched sample, 2000):")
    if "group" in pre_matched.columns:
        print(pre_matched.groupby(["county_name", "group"])["geoid"]
              .nunique().to_string())


def _smd(treated: pd.Series, control: pd.Series) -> float:
    pooled_sd = np.sqrt((treated.std()**2 + control.std()**2) / 2)
    return 0.0 if pooled_sd == 0 else (treated.mean() - control.mean()) / pooled_sd


# ── PARALLEL TRENDS CHECK ─────────────────────────────────────────────────────
# With 2000 as a clean pre-treatment baseline, we can test whether
# treated and control groups had similar housing values before any DC arrived.

def parallel_trends_check(panel: pd.DataFrame):
    if "group" not in panel.columns:
        print("WARNING: group column missing — run Script 3 first")
        return

    print(f"\n{'='*60}")
    print("PARALLEL TRENDS CHECK — values by period and group")
    print(f"{'='*60}")

    for outcome in ["med_home_value", "med_rent"]:
        if outcome not in panel.columns:
            continue
        print(f"\n  {outcome}:")
        for year in YEARS:
            yr = panel[panel["year"] == year]
            for group in ["control", "pre2000", "buildout", "hyperscale"]:
                vals = yr[yr["group"] == group][outcome].dropna()
                if len(vals) > 0:
                    print(f"    {year} {group:<12}: "
                          f"${vals.mean():>10,.0f}  (n={len(vals)})")

        # Key test: were buildout tracts similar to control in 2000?
        pre      = panel[panel["year"] == 2000]
        ctrl     = pre[pre["group"] == "control"][outcome].dropna()
        buildout = pre[pre["group"] == "buildout"][outcome].dropna()
        if len(ctrl) > 0 and len(buildout) > 0:
            _, p = stats.ttest_ind(buildout, ctrl, equal_var=False)
            print(f"\n  Buildout vs Control in 2000: p={p:.3f} "
                  f"{'← similar at baseline ✓' if p >= 0.05 else '← pre-existing difference, note in paper'}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    full_panel    = load_panel()
    parallel_trends_check(full_panel)
    matched_panel = match_within_county(full_panel)
    balance_table(full_panel, matched_panel)

    os.makedirs("data", exist_ok=True)
    matched_panel.to_csv("data/matched_panel.csv", index=False)
    print(f"\nMatched panel saved: data/matched_panel.csv")
    print("Ready to run Script 5 — fixed_effects_regression.py")


if __name__ == "__main__":
    import os
    main()