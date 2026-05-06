"""
Script 4b — Bias Reduction (FHFA Robustness Check)
====================================================
Identical to Script 4 but reads from the FHFA GeoPackage and
saves matched panel to data/matched_panel_fhfa.csv.
The primary Census matched_panel.csv is never touched.

Run after Script 3b — assign_treatment_groups_fhfa.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# ── FHFA-specific paths ────────────────────────────────────────────────────────
GPKG_PATH      = "geopackages/housing_data_fhfa.gpkg"
OUTPUT_CSV     = "data/matched_panel_fhfa.csv"
DEFAULT_BUFFER = 8
YEARS          = [2000, 2015, 2023]

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

CALIPER = 0.10


def load_panel() -> pd.DataFrame:
    gdf = gpd.read_file(GPKG_PATH, layer="study_tracts")

    print("\nColumns in FHFA study_tracts layer:")
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

        treatment_cols = [c for c in gdf.columns if (
            c.startswith("near_dc_")         or
            c.startswith("dist_nearest_dc_") or
            c in ["group_8km", "is_buildout_8km"]
        )]

        yr_df = gdf[["geoid", "county_name"] +
                    treatment_cols +
                    list(available.keys())].copy()
        yr_df = yr_df.rename(columns=available)
        yr_df["year"] = year
        rows.append(yr_df)

    panel = pd.concat(rows, ignore_index=True)
    if "geometry" in panel.columns:
        panel = panel.drop(columns=["geometry"])

    if "group_8km" in panel.columns:
        panel = panel.rename(columns={
            "group_8km":       "group",
            "is_buildout_8km": "is_buildout",
        })

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

    panel["data_source"] = "FHFA_HPI"

    print(f"FHFA Panel: {len(panel)} rows | "
          f"{panel['geoid'].nunique()} tracts | 3 periods")

    print("\n2000 covariate availability:")
    yr2000 = panel[panel["year"] == 2000]
    for col in PSM_COVARIATES:
        if col in yr2000.columns:
            n_valid = yr2000[col].notna().sum()
            print(f"  {col}: {n_valid} non-null out of {len(yr2000)}")
        else:
            print(f"  {col}: not found")

    if "group" in panel.columns:
        print(f"\nTract counts by county and group at {DEFAULT_BUFFER}km (2000):")
        print(panel[panel["year"] == 2000].groupby(
            ["county_name", "group"])["geoid"].nunique().to_string())

    return panel


def match_within_county(panel: pd.DataFrame,
                        buffer_km: int = DEFAULT_BUFFER) -> pd.DataFrame:
    print(f"\nRunning PSM within each county (buffer = {buffer_km}km)...")
    matched_frames = []

    panel = panel.copy()
    if "dist_nearest_dc_2023_km" in panel.columns:
        panel["group"]       = np.where(
            panel["dist_nearest_dc_2023_km"] <= buffer_km,
            "buildout", "control"
        )
        panel["is_buildout"] = (panel["group"] == "buildout").astype(int)
        panel["buildout_x_post_2015"] = (
            panel["is_buildout"] * panel["post_2015"]
        )
        panel["buildout_x_post_2023"] = (
            panel["is_buildout"] * panel["post_2023"]
        )

    for county in STUDY_COUNTIES:
        sub = panel[
            (panel["county_name"] == county) &
            (panel["year"] == 2000)
        ].copy()

        available_covs = [c for c in PSM_COVARIATES if c in sub.columns]
        if len(available_covs) < 2:
            print(f"  {county}: not enough covariates — skipping")
            continue

        if "is_buildout" not in sub.columns:
            print(f"  {county}: treatment columns missing — run Script 3b first")
            continue

        clean   = sub.dropna(subset=available_covs + ["is_buildout"])
        treated = clean[clean["is_buildout"] == 1]
        control = clean[clean["group"] == "control"]

        if len(treated) == 0 or len(control) == 0:
            print(f"  {county}: no treated or control tracts — skipping")
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

        n_dropped = (~valid).sum()
        print(f"  {county}: {len(matched_treat)} matched pairs "
              f"({n_dropped} outside caliper)")

        matched_frames.append(
            panel[(panel["county_name"] == county) &
                  (panel["geoid"].isin(matched_geoids))]
        )

    if not matched_frames:
        print("WARNING: No matched samples produced")
        return panel

    matched = pd.concat(matched_frames, ignore_index=True)
    print(f"\nFHFA matched panel: {matched['geoid'].nunique()} tracts "
          f"across {matched['county_name'].nunique()} counties")
    print(f"\nMatched tract counts by county and group (2000 baseline):")
    print(matched[matched["year"] == 2000].groupby(
        ["county_name", "group"])["geoid"].nunique().to_string())
    return matched


def balance_table(full_panel: pd.DataFrame, matched_panel: pd.DataFrame):
    covariates = [c for c in PSM_COVARIATES if c in full_panel.columns]
    if not covariates:
        return

    print(f"\n{'='*75}")
    print("FHFA BALANCE TABLE — 2000 baseline covariates")
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


def _smd(treated: pd.Series, control: pd.Series) -> float:
    pooled_sd = np.sqrt((treated.std()**2 + control.std()**2) / 2)
    return 0.0 if pooled_sd == 0 else (treated.mean() - control.mean()) / pooled_sd


def parallel_trends_check(panel: pd.DataFrame):
    if "group" not in panel.columns:
        return

    print(f"\n{'='*60}")
    print("FHFA PARALLEL TRENDS CHECK")
    print(f"{'='*60}")

    for outcome in ["med_home_value", "med_rent"]:
        if outcome not in panel.columns:
            continue
        print(f"\n  {outcome}:")
        for year in YEARS:
            yr = panel[panel["year"] == year]
            for group in ["control", "buildout"]:
                vals = yr[yr["group"] == group][outcome].dropna()
                if len(vals) > 0:
                    print(f"    {year} {group:<12}: "
                          f"${vals.mean():>10,.0f}  (n={len(vals)})")

        pre      = panel[panel["year"] == 2000]
        ctrl     = pre[pre["group"] == "control"][outcome].dropna()
        buildout = pre[pre["group"] == "buildout"][outcome].dropna()
        if len(ctrl) > 0 and len(buildout) > 0:
            _, p = stats.ttest_ind(buildout, ctrl, equal_var=False)
            print(f"\n  Buildout vs Control in 2000: p={p:.3f} "
                  f"{'← similar at baseline ✓' if p >= 0.05 else '← pre-existing difference, note in paper'}")


def main():
    print("Script 4b — FHFA Robustness Check")
    print(f"Reading from:  {GPKG_PATH}")
    print(f"Saving to:     {OUTPUT_CSV}")

    full_panel    = load_panel()
    parallel_trends_check(full_panel)
    matched_panel = match_within_county(full_panel, buffer_km=DEFAULT_BUFFER)
    balance_table(full_panel, matched_panel)

    os.makedirs("data", exist_ok=True)
    matched_panel.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFHFA matched panel saved: {OUTPUT_CSV}")
    print("Ready to run Script 5b — fixed_effects_regression_fhfa.py")


if __name__ == "__main__":
    main()
