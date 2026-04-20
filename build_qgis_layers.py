"""
Script 7 — QGIS Buffer Layer: Home Value Change vs County Control
==================================================================
Builds a GeoPackage layer for QGIS showing how much more or less
each buffer ring around a data center appreciated compared to
non-DC tracts in the same county.

Primary metric per ring:
  hv_pct_vs_control  =  (ring_chg_2000_2023 - county_control_chg)
                         / abs(county_control_chg) * 100

  Positive = ring appreciated MORE than non-DC county tracts
  Negative = ring appreciated LESS than non-DC county tracts

Control group: all tracts in the same county where
  dist_nearest_dc_2023_km > 8km (unmatched, county-wide)

This is calculated separately for each county so that Sacramento
is compared to Sacramento non-DC tracts, SF to SF non-DC tracts,
etc. Counties are never pooled.

For each DC, four concentric rings are created:
  0–2km   (immediate vicinity)
  2–4km   (near neighborhood)
  4–6km   (mid-range)
  6–8km   (outer influence zone)

Outputs (written to geopackages/housing_data.gpkg):
  dc_buffer_rings        — rings with vs-control pct difference
  dc_points_study        — DC point locations for overlay
  tract_home_value_change — tract polygons with full change data

Run after Scripts 3 and 4 are complete.

Install: pip install pandas geopandas shapely
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")

# ── SETTINGS ──────────────────────────────────────────────────────────────────
GPKG_PATH      = "geopackages/housing_data.gpkg"
PANEL_CSV      = "data/matched_panel.csv"
RAW_DC_CSV     = "im3_open_source_data_center_atlas_v2026.02.09.csv"
EPSG_PROJ      = 3310
CONTROL_MIN_KM = 8.0   # tracts beyond this distance are the control group

RING_BOUNDS = [
    (0,    2000, "0–2km"),
    (2000, 4000, "2–4km"),
    (4000, 6000, "4–6km"),
    (6000, 8000, "6–8km"),
]

STUDY_COUNTIES = [
    "Sacramento County",
    "Los Angeles County",
    "Alameda County",
    "Napa County",
    "San Francisco County",
]

from opening_year_lookup import (
    load_study_counties,
    build_dated_gdf,
)


# ── LOAD DC LOCATIONS ─────────────────────────────────────────────────────────

def load_study_dcs() -> gpd.GeoDataFrame:
    study     = load_study_counties(RAW_DC_CSV)
    dated_dcs = build_dated_gdf(study)
    print(f"Loaded {len(dated_dcs)} study DCs")
    return dated_dcs


# ── LOAD TRACT LAYER ──────────────────────────────────────────────────────────

def load_tract_layer() -> gpd.GeoDataFrame:
    tracts = gpd.read_file(GPKG_PATH, layer="study_tracts")
    if tracts.crs.to_epsg() != EPSG_PROJ:
        tracts = tracts.to_crs(epsg=EPSG_PROJ)
    print(f"Loaded {len(tracts)} tracts from GeoPackage")
    return tracts


# ── ATTACH HOUSING CHANGE TO TRACTS ──────────────────────────────────────────

def attach_housing_change(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    panel = pd.read_csv(PANEL_CSV, dtype={"geoid": str})

    for year in [2000, 2015, 2023]:
        yr = panel[panel["year"] == year][
            ["geoid", "med_home_value", "med_rent", "med_hh_income"]
        ].rename(columns={
            "med_home_value": f"hv_{year}",
            "med_rent":       f"rent_{year}",
            "med_hh_income":  f"inc_{year}",
        })
        tracts = tracts.merge(yr, on="geoid", how="left")

    # Absolute change
    for col, prefix in [("hv", "home_value"), ("rent", "rent")]:
        for y0, y1 in [(2000, 2015), (2015, 2023), (2000, 2023)]:
            tracts[f"{prefix}_chg_{y0}_{y1}"] = (
                tracts[f"{col}_{y1}"] - tracts[f"{col}_{y0}"]
            )

    # Percent change on tracts for reference layer
    for col, prefix in [("hv", "home_value"), ("rent", "rent")]:
        for y0, y1 in [(2000, 2015), (2015, 2023), (2000, 2023)]:
            tracts[f"{prefix}_pct_{y0}_{y1}"] = (
                (tracts[f"{col}_{y1}"] - tracts[f"{col}_{y0}"]) /
                tracts[f"{col}_{y0}"] * 100
            ).round(2)

    print(f"  Housing change columns attached to {len(tracts)} tracts")
    return tracts


# ── COMPUTE COUNTY CONTROL BENCHMARKS ────────────────────────────────────────
# For each county, computes the mean home value and rent change for
# all tracts that are more than 8km from any DC — these are the
# county-level control benchmarks each ring is compared against.

def compute_county_benchmarks(tracts: gpd.GeoDataFrame) -> dict:
    print("\nComputing county control benchmarks (tracts > 8km from any DC)...")
    benchmarks = {}

    for county in STUDY_COUNTIES:
        county_mask = tracts["county_name"].str.contains(
            county.replace(" County", ""), na=False
        )
        county_tracts = tracts[county_mask].copy()

        if len(county_tracts) == 0:
            continue

        # Control tracts = beyond 8km from any DC
        if "dist_nearest_dc_2023_km" in county_tracts.columns:
            control_mask = county_tracts["dist_nearest_dc_2023_km"] > CONTROL_MIN_KM
            control_tracts = county_tracts[control_mask]
        else:
            # Fallback if distance column missing — use all county tracts
            print(f"  WARNING: dist column not found for {county}, "
                  f"using all county tracts as control")
            control_tracts = county_tracts

        n_control = len(control_tracts)

        benchmarks[county] = {}
        for outcome, col in [("home_value", "home_value_chg_2000_2023"),
                              ("home_value_2015", "home_value_chg_2000_2015"),
                              ("home_value_late", "home_value_chg_2015_2023"),
                              ("rent", "rent_chg_2000_2023")]:
            if col in control_tracts.columns:
                vals = control_tracts[col].dropna()
                benchmarks[county][outcome] = {
                    "mean":    vals.mean() if len(vals) > 0 else np.nan,
                    "n":       len(vals),
                    "n_total": n_control,
                }
            else:
                benchmarks[county][outcome] = {
                    "mean": np.nan, "n": 0, "n_total": n_control
                }

        ctrl_chg = benchmarks[county]["home_value"]["mean"]
        print(f"  {county:<25} "
              f"{n_control:>4} control tracts  |  "
              f"control hv chg 2000→2023: "
              f"${ctrl_chg:>10,.0f}" if not np.isnan(ctrl_chg)
              else f"  {county:<25} {n_control:>4} control tracts  |  "
                   f"control hv chg: n/a")

    return benchmarks


# ── BUILD RING LAYER ──────────────────────────────────────────────────────────
# For each DC ring, computes mean home value change and expresses it
# as a percent difference from the county control benchmark.

def build_ring_layer(dcs:        gpd.GeoDataFrame,
                     tracts:     gpd.GeoDataFrame,
                     benchmarks: dict) -> gpd.GeoDataFrame:
    print("\nBuilding buffer rings with vs-control comparison...")
    rings = []

    tract_centroids             = tracts.copy()
    tract_centroids["centroid"] = tracts.geometry.centroid

    for _, dc in dcs.iterrows():
        dc_point  = dc.geometry
        dc_county = dc.get("county", "")

        if dc_county not in STUDY_COUNTIES:
            continue
        if dc_county not in benchmarks:
            print(f"  Skipping {dc_county} — no benchmark computed")
            continue

        bench = benchmarks[dc_county]

        for inner_m, outer_m, label in RING_BOUNDS:
            outer_circle = dc_point.buffer(outer_m)
            inner_circle = dc_point.buffer(inner_m) if inner_m > 0 else None
            ring_geom    = (outer_circle.difference(inner_circle)
                            if inner_circle else outer_circle)

            in_ring = tract_centroids[
                tract_centroids["centroid"].within(ring_geom)
            ]

            def smean(col):
                vals = (in_ring[col].dropna()
                        if col in in_ring.columns else pd.Series())
                return round(float(vals.mean()), 2) if len(vals) > 0 else np.nan

            # Mean change for tracts in this ring
            ring_hv_chg        = smean("home_value_chg_2000_2023")
            ring_hv_chg_early  = smean("home_value_chg_2000_2015")
            ring_hv_chg_late   = smean("home_value_chg_2015_2023")
            ring_rent_chg      = smean("rent_chg_2000_2023")

            # County control benchmarks
            ctrl_hv_chg       = bench["home_value"]["mean"]
            ctrl_hv_chg_early = bench["home_value_2015"]["mean"]
            ctrl_hv_chg_late  = bench["home_value_late"]["mean"]
            ctrl_rent_chg     = bench["rent"]["mean"]
            n_control         = bench["home_value"]["n_total"]

            # ── PRIMARY METRIC: % difference vs county control ──
            def pct_vs_ctrl(ring_val, ctrl_val):
                if np.isnan(ring_val) or np.isnan(ctrl_val) or ctrl_val == 0:
                    return np.nan
                return round((ring_val - ctrl_val) / abs(ctrl_val) * 100, 2)

            hv_pct_vs_ctrl      = pct_vs_ctrl(ring_hv_chg, ctrl_hv_chg)
            hv_pct_vs_ctrl_early = pct_vs_ctrl(ring_hv_chg_early,
                                                ctrl_hv_chg_early)
            hv_pct_vs_ctrl_late  = pct_vs_ctrl(ring_hv_chg_late,
                                                ctrl_hv_chg_late)
            rent_pct_vs_ctrl    = pct_vs_ctrl(ring_rent_chg, ctrl_rent_chg)

            ring_record = {
                # DC identifiers
                "dc_id":        dc.get("id", ""),
                "dc_name":      str(dc.get("name", "")),
                "operator":     str(dc.get("operator", "")),
                "county":       dc_county,
                "sqft":         float(dc.get("sqft", np.nan)),
                "opening_year": int(dc.get("opening_year", 0)),
                "ring_label":   label,
                "inner_km":     inner_m / 1000,
                "outer_km":     outer_m / 1000,
                "n_tracts":     len(in_ring),
                "n_control":    n_control,

                # ── PRIMARY FIELDS FOR QGIS ──
                # Positive = ring outperformed county non-DC tracts
                # Negative = ring underperformed county non-DC tracts
                "hv_pct_vs_ctrl":       hv_pct_vs_ctrl,
                "hv_pct_vs_ctrl_early": hv_pct_vs_ctrl_early,
                "hv_pct_vs_ctrl_late":  hv_pct_vs_ctrl_late,
                "rent_pct_vs_ctrl":     rent_pct_vs_ctrl,

                # ── RAW VALUES FOR REFERENCE ──
                "ring_hv_chg_2000_2023":  ring_hv_chg,
                "ctrl_hv_chg_2000_2023":  round(float(ctrl_hv_chg), 2)
                                           if not np.isnan(ctrl_hv_chg)
                                           else np.nan,
                "ring_rent_chg_2000_2023": ring_rent_chg,
                "ctrl_rent_chg_2000_2023": round(float(ctrl_rent_chg), 2)
                                            if not np.isnan(ctrl_rent_chg)
                                            else np.nan,
                "mean_hv_2000":  smean("hv_2000"),
                "mean_hv_2023":  smean("hv_2023"),

                "geometry": ring_geom,
            }
            rings.append(ring_record)

    ring_gdf = gpd.GeoDataFrame(rings, crs=f"EPSG:{EPSG_PROJ}")
    return ring_gdf


# ── PRINT COUNTY SUMMARY TABLE ────────────────────────────────────────────────
# Shows the vs-control % difference by ring and county.
# Positive = ring outperformed non-DC county tracts.

def print_county_summary(ring_gdf: gpd.GeoDataFrame,
                         benchmarks: dict):
    print(f"\n{'='*80}")
    print("HOME VALUE CHANGE VS COUNTY CONTROL — % DIFFERENCE BY RING (2000→2023)")
    print("Positive = ring appreciated MORE than non-DC county tracts")
    print("Negative = ring appreciated LESS than non-DC county tracts")
    print(f"{'='*80}")
    print(f"  {'County':<22} {'DC':<28} "
          f"{'0–2km':>9} {'2–4km':>9} {'4–6km':>9} {'6–8km':>9} "
          f"{'Control $':>12}")
    print("  " + "─" * 80)

    for county in STUDY_COUNTIES:
        county_rings = ring_gdf[ring_gdf["county"] == county]
        if county_rings.empty:
            continue

        ctrl_chg = (benchmarks.get(county, {})
                    .get("home_value", {}).get("mean", np.nan))
        ctrl_str = (f"${ctrl_chg:>10,.0f}"
                    if not np.isnan(ctrl_chg) else "n/a")

        for dc_name in county_rings["dc_name"].unique():
            dc_rings = county_rings[
                county_rings["dc_name"] == dc_name
            ].sort_values("inner_km")

            row = []
            for label in ["0–2km", "2–4km", "4–6km", "6–8km"]:
                ring = dc_rings[dc_rings["ring_label"] == label]
                if len(ring) > 0:
                    val = ring["hv_pct_vs_ctrl"].values[0]
                    if np.isnan(val):
                        row.append("      n/a")
                    else:
                        sign = "+" if val > 0 else ""
                        row.append(f"{sign}{val:>7.1f}%")
                else:
                    row.append("      n/a")

            short_dc = str(dc_name)[:27]
            print(f"  {county:<22} {short_dc:<28} "
                  f"{'  '.join(row)}  {ctrl_str}")

    print(f"\n{'='*80}")
    print("RENT CHANGE VS COUNTY CONTROL — % DIFFERENCE BY RING (2000→2023)")
    print(f"{'='*80}")
    print(f"  {'County':<22} {'DC':<28} "
          f"{'0–2km':>9} {'2–4km':>9} {'4–6km':>9} {'6–8km':>9} "
          f"{'Control $':>12}")
    print("  " + "─" * 80)

    for county in STUDY_COUNTIES:
        county_rings = ring_gdf[ring_gdf["county"] == county]
        if county_rings.empty:
            continue

        ctrl_rent = (benchmarks.get(county, {})
                     .get("rent", {}).get("mean", np.nan))
        ctrl_str  = (f"${ctrl_rent:>10,.0f}"
                     if not np.isnan(ctrl_rent) else "n/a")

        for dc_name in county_rings["dc_name"].unique():
            dc_rings = county_rings[
                county_rings["dc_name"] == dc_name
            ].sort_values("inner_km")

            row = []
            for label in ["0–2km", "2–4km", "4–6km", "6–8km"]:
                ring = dc_rings[dc_rings["ring_label"] == label]
                if len(ring) > 0:
                    val = ring["rent_pct_vs_ctrl"].values[0]
                    if np.isnan(val):
                        row.append("      n/a")
                    else:
                        sign = "+" if val > 0 else ""
                        row.append(f"{sign}{val:>7.1f}%")
                else:
                    row.append("      n/a")

            short_dc = str(dc_name)[:27]
            print(f"  {county:<22} {short_dc:<28} "
                  f"{'  '.join(row)}  {ctrl_str}")

    print(f"\n  Control = all tracts in county with dist_nearest_dc > 8km")
    print(f"  n/a = no tracts found in that ring")


# ── BUILD COUNTY TWFE LAYER ───────────────────────────────────────────────────
# Creates a county polygon layer attributed with the TWFE regression
# coefficients from Script 5. This is the regression-adjusted causal
# estimate displayed at the county level alongside the descriptive rings.
#
# Values are hardcoded from Script 5 output (8km primary specification).
# Update these if you rerun Script 5 with different settings.
#
# County-by-county TWFE results (buildout × post2023):
#   Sacramento:    β = +3,565    p = 0.940  n.s.
#   Los Angeles:   β = +48,769   p = 0.378  n.s.
#   Alameda:       β = +91,971   p = 0.594  n.s.
#   San Francisco: β = +64,004   p = 0.384  n.s.
#   Pooled TWFE:   β = +123,307  p = 0.008  ***
#
# Note: county-level models are underpowered (small matched sample).
# The pooled TWFE is the primary causal estimate — shown as an
# annotation on each county polygon for interpretive context.

TWFE_BY_COUNTY = {
    "Sacramento County": {
        "twfe_coef_hv":    3565.03,
        "twfe_se_hv":      46000.00,   # approximate — county model underpowered
        "twfe_p_hv":       0.940,
        "twfe_sig_hv":     "n.s.",
        "twfe_coef_rent":  -63.81,
        "twfe_p_rent":     0.426,
        "twfe_sig_rent":   "n.s.",
    },
    "Los Angeles County": {
        "twfe_coef_hv":    48768.50,
        "twfe_se_hv":      54000.00,
        "twfe_p_hv":       0.378,
        "twfe_sig_hv":     "n.s.",
        "twfe_coef_rent":  103.37,
        "twfe_p_rent":     0.213,
        "twfe_sig_rent":   "n.s.",
    },
    "Alameda County": {
        "twfe_coef_hv":    91971.43,
        "twfe_se_hv":      162000.00,
        "twfe_p_hv":       0.594,
        "twfe_sig_hv":     "n.s.",
        "twfe_coef_rent":  26.71,
        "twfe_p_rent":     0.900,
        "twfe_sig_rent":   "n.s.",
    },
    "San Francisco County": {
        "twfe_coef_hv":    64004.43,
        "twfe_se_hv":      72000.00,
        "twfe_p_hv":       0.384,
        "twfe_sig_hv":     "n.s.",
        "twfe_coef_rent":  53.22,
        "twfe_p_rent":     0.693,
        "twfe_sig_rent":   "n.s.",
    },
}

# Pooled TWFE result — shown as context on all county polygons
POOLED_TWFE = {
    "pooled_twfe_coef_hv":   123306.97,
    "pooled_twfe_se_hv":     46097.27,
    "pooled_twfe_p_hv":      0.008,
    "pooled_twfe_sig_hv":    "***",
    "pooled_twfe_coef_rent": 133.90,
    "pooled_twfe_se_hv":     67.16,
    "pooled_twfe_p_rent":    0.047,
    "pooled_twfe_sig_rent":  "**",
}


def build_county_twfe_layer(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Dissolves tract polygons to county boundaries and attributes each
    county with TWFE regression results from Script 5.
    The primary field for QGIS labeling is twfe_coef_hv (home value)
    and twfe_sig_hv (significance stars).
    """
    print("\nBuilding county TWFE layer...")

    # Dissolve tracts to county boundaries
    county_polys = (
        tracts[["county_name", "geometry"]]
        .dissolve(by="county_name")
        .reset_index()
    )

    # Attach TWFE results
    rows = []
    for _, row in county_polys.iterrows():
        county = row["county_name"]
        twfe   = TWFE_BY_COUNTY.get(county, {})

        record = {
            "county_name":       county,
            "geometry":          row["geometry"],

            # County-level TWFE (underpowered — for reference)
            "twfe_coef_hv":      twfe.get("twfe_coef_hv",   np.nan),
            "twfe_se_hv":        twfe.get("twfe_se_hv",     np.nan),
            "twfe_p_hv":         twfe.get("twfe_p_hv",      np.nan),
            "twfe_sig_hv":       twfe.get("twfe_sig_hv",    ""),
            "twfe_coef_rent":    twfe.get("twfe_coef_rent", np.nan),
            "twfe_p_rent":       twfe.get("twfe_p_rent",    np.nan),
            "twfe_sig_rent":     twfe.get("twfe_sig_rent",  ""),

            # Pooled TWFE — primary causal estimate
            "pooled_coef_hv":    POOLED_TWFE["pooled_twfe_coef_hv"],
            "pooled_p_hv":       POOLED_TWFE["pooled_twfe_p_hv"],
            "pooled_sig_hv":     POOLED_TWFE["pooled_twfe_sig_hv"],
            "pooled_coef_rent":  POOLED_TWFE["pooled_twfe_coef_rent"],
            "pooled_p_rent":     POOLED_TWFE["pooled_twfe_p_rent"],
            "pooled_sig_rent":   POOLED_TWFE["pooled_twfe_sig_rent"],

            # Human-readable label for QGIS map annotation
            "label_hv": (
                f"{county.replace(' County','')}\n"
                f"County TWFE: ${twfe.get('twfe_coef_hv', 0):+,.0f} "
                f"{twfe.get('twfe_sig_hv','')}\n"
                f"Pooled TWFE: $+123,307 ***"
            ),
            "label_rent": (
                f"{county.replace(' County','')}\n"
                f"County TWFE: ${twfe.get('twfe_coef_rent', 0):+,.0f} "
                f"{twfe.get('twfe_sig_rent','')}\n"
                f"Pooled TWFE: $+134 **"
            ),
        }
        rows.append(record)

    county_gdf = gpd.GeoDataFrame(rows, crs=f"EPSG:{EPSG_PROJ}")

    print(f"  Built {len(county_gdf)} county polygons with TWFE attributes")
    for _, row in county_gdf.iterrows():
        print(f"  {row['county_name']:<25} "
              f"TWFE hv: ${row['twfe_coef_hv']:>+10,.0f} "
              f"{row['twfe_sig_hv']:<6}  "
              f"rent: ${row['twfe_coef_rent']:>+8,.0f} "
              f"{row['twfe_sig_rent']}")

    return county_gdf


# ── BUILD DC POINT LAYER ──────────────────────────────────────────────────────

def build_dc_points(dcs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols      = ["id", "name", "operator", "county", "sqft",
                 "opening_year", "geometry"]
    available = [c for c in cols if c in dcs.columns]
    return dcs[available].copy()


# ── BUILD CALIFORNIA COUNTY BOUNDARIES ───────────────────────────────────────
# Downloads CA county boundaries from Census TIGER files (same source
# as Script 2's tract boundaries). Produces two layers:
#   california_counties_all  — all 58 CA counties, light fill for context
#   california_counties_study — only the four study counties, highlighted
#
# The all-counties layer gives readers geographic orientation.
# The study layer draws attention to the specific counties analyzed.

def build_california_counties(tracts: gpd.GeoDataFrame) -> tuple:
    import requests
    import zipfile

    cache       = "data/tl_2020_us_county.zip"
    extract_dir = "data/tl_2020_us_county"
    os.makedirs("data", exist_ok=True)

    print("\nBuilding California county boundaries layer...")

    # Validate existing zip — delete and re-download if corrupted
    if os.path.exists(cache):
        try:
            with zipfile.ZipFile(cache, "r") as z:
                z.namelist()
        except zipfile.BadZipFile:
            print("  Cached zip is corrupted — re-downloading...", end=" ")
            os.remove(cache)
            if os.path.exists(extract_dir):
                import shutil
                shutil.rmtree(extract_dir)

    if not os.path.exists(cache):
        # National county file — filter to CA after loading
        url = ("https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/"
               "tl_2020_us_county.zip")
        print("  Downloading US county boundaries (~75 MB)...", end=" ")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(cache, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print("done")

    # Extract to folder if not already done
    if not os.path.exists(extract_dir):
        print("  Extracting county boundaries...", end=" ")
        with zipfile.ZipFile(cache, "r") as z:
            z.extractall(extract_dir)
        print("done")
    else:
        print("  Using cached county boundaries")

    # Find the .shp file inside the extracted folder
    shp_files = [f for f in os.listdir(extract_dir) if f.endswith(".shp")]
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {extract_dir}")
    shp_path = os.path.join(extract_dir, shp_files[0])

    counties_all = (
        gpd.read_file(shp_path)
        .to_crs(epsg=EPSG_PROJ)
    )

    # Filter to California only (STATEFP = "06")
    counties_all = counties_all[counties_all["STATEFP"] == "06"].copy()
    print(f"  Filtered to {len(counties_all)} California counties")
    counties_all = counties_all.rename(columns={"NAME": "county_name"})
    counties_all["county_name"] = counties_all["county_name"] + " County"

    # Flag study counties
    counties_all["is_study_county"] = (
        counties_all["county_name"].isin(STUDY_COUNTIES)
    ).astype(int)

    # Study counties subset — highlighted layer
    counties_study = counties_all[
        counties_all["is_study_county"] == 1
    ].copy()

    print(f"  Loaded {len(counties_all)} CA counties")
    print(f"  Study counties: {len(counties_study)} highlighted")

    return counties_all, counties_study


# ── EXPORT TO GEOPACKAGE ──────────────────────────────────────────────────────

def export_to_gpkg(ring_layer:      gpd.GeoDataFrame,
                   dc_points:       gpd.GeoDataFrame,
                   tract_layer:     gpd.GeoDataFrame,
                   county_layer:    gpd.GeoDataFrame,
                   ca_all:          gpd.GeoDataFrame,
                   ca_study:        gpd.GeoDataFrame):
    print(f"\nExporting to {GPKG_PATH}...")
    layers = {
        "dc_buffer_rings":          ring_layer,
        "dc_points_study":          dc_points,
        "tract_home_value_change":  tract_layer,
        "county_twfe_results":      county_layer,
        "california_counties_all":  ca_all,
        "california_counties_study": ca_study,
    }
    for name, gdf in layers.items():
        if gdf is None or len(gdf) == 0:
            print(f"  Skipped {name} (empty)")
            continue
        drop = [c for c in gdf.columns if c.startswith("index_")]
        gdf.drop(columns=drop, errors="ignore").to_file(
            GPKG_PATH, layer=name, driver="GPKG"
        )
        print(f"  ✓ {name} ({len(gdf)} features)")
    print(f"\nGeoPackage updated: {GPKG_PATH}")


# ── QGIS STYLING GUIDE ────────────────────────────────────────────────────────

def print_qgis_guide():
    print(f"""
{'='*65}
QGIS STYLING GUIDE
{'='*65}

PRIMARY VISUALIZATION LAYER: dc_buffer_rings
─────────────────────────────────────────────
  Primary field:  hv_pct_vs_ctrl
  Interpretation: % by which this ring outperformed (+) or
                  underperformed (-) county non-DC tracts

  Renderer:  Graduated color
  Color ramp: Diverging — Red → White → Green
              Red   = negative (underperformed control)
              White = 0 (same as control)
              Green = positive (outperformed control)
  Classes:   5, centered on 0
  Mode:      Equal interval or manual breaks centered at 0
  Opacity:   70%

  All field descriptions:
    hv_pct_vs_ctrl        — PRIMARY: full period vs control (%)
    hv_pct_vs_ctrl_early  — 2000→2015 vs control (%)
    hv_pct_vs_ctrl_late   — 2015→2023 vs control (%)
    rent_pct_vs_ctrl      — rent full period vs control (%)
    ring_hv_chg_2000_2023 — raw $ change in this ring
    ctrl_hv_chg_2000_2023 — county control $ change (same for all rings)
    n_tracts              — tracts in this ring
    n_control             — county control tracts used as benchmark
    ring_label            — distance zone
    county                — county (filter by this for per-county maps)
    dc_name               — which DC this ring surrounds
    opening_year          — when DC opened

  Tip: Set manual color ramp breaks at:
    -50%, -25%, 0%, +25%, +50%
  This ensures the zero line is always white and the
  map reads intuitively regardless of the data range.

REFERENCE LAYER: tract_home_value_change
─────────────────────────────────────────
  Field:    home_value_pct_2000_2023
  Renderer: Sequential blue
  Opacity:  35%

DC POINTS: dc_points_study
───────────────────────────
  Symbol: Star, size 8, color #0F766E
  Label:  dc_name + opening_year

LAYER ORDER:
  1. dc_points_study
  2. dc_buffer_rings
  3. county_twfe_results
  4. tract_home_value_change
  5. california_counties_study
  6. california_counties_all
  7. Basemap (optional — may not be needed with county layer)

CALIFORNIA COUNTY LAYERS
──────────────────────────
  california_counties_all:
    Renderer:  Single symbol
    Fill:      Light grey #F3F4F6, opacity 60%
    Outline:   Medium grey #9CA3AF, width 0.3mm
    Labels:    county_name field, size 7, grey #6B7280
    Use as:    Geographic orientation layer showing all 58 counties

  california_counties_study:
    Renderer:  Single symbol
    Fill:      No fill (transparent)
    Outline:   Dark teal #0F766E, width 0.8mm, dashed
    Labels:    county_name field, size 9, bold, teal #0F766E
               with white halo size 1.5
    Use as:    Highlights the four study counties so readers
               immediately know which areas are being analyzed

  For the statewide overview map:
    Show california_counties_all + california_counties_study
    + dc_points_study only — no rings or tract layer
    This gives a clean locator map showing where the DCs are

  For per-county close-up maps:
    Turn off california_counties_all
    Keep california_counties_study as a border reference

COUNTY TWFE LAYER: county_twfe_results
────────────────────────────────────────
  This layer shows the regression-adjusted causal estimate
  from Script 5 at the county level. Use it as an annotation
  overlay so readers can see both the descriptive ring patterns
  AND the regression-adjusted estimate on the same map.

  Renderer:  No fill (transparent), outline only
  Outline:   Dark grey, width 0.5mm
  Opacity:   100%
  Labels:    Use label_hv field for home value maps
             Use label_rent field for rent maps
  Font:      Bold, size 8, black with white halo

  Field descriptions:
    twfe_coef_hv     — county-level TWFE β for home value
    twfe_sig_hv      — significance stars (county model)
    twfe_coef_rent   — county-level TWFE β for rent
    pooled_coef_hv   — pooled TWFE β (+$123,307 ***)
    pooled_sig_hv    — pooled significance (***)
    pooled_coef_rent — pooled TWFE β for rent (+$134 **)
    label_hv         — pre-formatted label for map annotation
    label_rent       — pre-formatted label for map annotation

  NOTE: County-level TWFE is underpowered (small matched sample)
  and all county results are n.s. The pooled_coef fields show
  the primary causal estimate which IS significant. Make this
  distinction clear in your map caption.

PER-COUNTY MAPS:
  Use Atlas in the Print Layout with dc_buffer_rings
  filtered by county field. This auto-generates one
  map page per county from a single layout template.
{'='*65}
""")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    dcs              = load_study_dcs()
    tracts           = load_tract_layer()
    tracts           = attach_housing_change(tracts)
    benchmarks       = compute_county_benchmarks(tracts)
    rings            = build_ring_layer(dcs, tracts, benchmarks)
    print_county_summary(rings, benchmarks)
    points           = build_dc_points(dcs)
    counties         = build_county_twfe_layer(tracts)
    ca_all, ca_study = build_california_counties(tracts)
    export_to_gpkg(rings, points, tracts, counties, ca_all, ca_study)
    print_qgis_guide()


if __name__ == "__main__":
    main()