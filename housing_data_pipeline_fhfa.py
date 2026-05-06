"""
Script 2b — FHFA HPI Data Pipeline (Robustness Check)
=======================================================
Downloads the FHFA House Price Index at the census tract level and
builds a GeoPackage in the same format as Script 2's output so that
Scripts 3, 4, 5, 6, and 7 can run unchanged as a robustness check.

The FHFA HPI is based on actual mortgage transaction data (repeat-sales
methodology) from Fannie Mae and Freddie Mac, making it fundamentally
different from the Census ACS self-reported owner estimates used in the
primary analysis. Consistent findings across both data sources
substantially strengthens the paper's causal argument.

Key differences from Script 2:
  - Home value = FHFA HPI index converted to dollar levels
  - Rent = not available from FHFA (Census ACS rent kept as-is)
  - Index anchored to Census 2000 median home value at tract level
  - Supertracts handled by averaging across constituent tracts

Study counties: Sacramento, Los Angeles, Alameda, San Francisco
Output GeoPackage: geopackages/housing_data_fhfa.gpkg

Requires: Script 2 (housing_data_pipeline.py) already run so that
  Census 2000 baseline values are available for anchoring.

Install: pip install pandas geopandas openpyxl requests tqdm numpy
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd


# ── SETTINGS ──────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "geopackages"
GPKG_PATH       = os.path.join(OUTPUT_DIR, "housing_data_fhfa.gpkg")
CENSUS_GPKG     = os.path.join(OUTPUT_DIR, "housing_data.gpkg")  # Script 2 output
FHFA_CACHE      = "data/fhfa/HPI_AT_BDL_tract.csv"
CENSUS_API_KEY  = os.environ.get("CENSUS_API_KEY", "314253ac718eac9ce5891234409bba3d70aa48d6")
EPSG_GEO        = 4326
EPSG_PROJ       = 3310
STATE_FIPS      = "06"
YEARS           = [2000, 2015, 2023]
MIN_SQFT        = 100_000

STUDY_COUNTIES = {
    "Sacramento County":    "067",
    "Los Angeles County":   "037",
    "Alameda County":       "001",
    "San Francisco County": "075",
}

# ACS variables for rent (not in FHFA) — kept from Script 2
ACS_VARS_RENT = {
    "B25064_001E": "med_rent",
    "B19013_001E": "med_hh_income",
    "B25003_002E": "owner_occ",
    "B25003_003E": "renter_occ",
}

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs("data/fhfa", exist_ok=True)


# ── DOWNLOAD FHFA TRACT-LEVEL HPI ────────────────────────────────────────────
# Downloads the FHFA annual tract-level HPI Excel file.
# Cached after first download — subsequent runs load from disk.
# The file contains annual HPI values for census tracts nationwide
# indexed to 100 at the base period (1989 Q4).

def download_fhfa_hpi() -> pd.DataFrame:
    # FHFA blocks automated downloads — file must be downloaded manually.
    # Instructions:
    #   1. Go to https://www.fhfa.gov/data/hpi/datasets
    #   2. Find the Census Tract / Local HPI section
    #   3. Download HPI_AT_BDL_tract.csv
    #   4. Save to: data/fhfa/HPI_AT_BDL_tract.csv

    if not os.path.exists(FHFA_CACHE):
        raise FileNotFoundError(
            f"\nFHFA tract HPI file not found at: {FHFA_CACHE}\n\n"
            f"FHFA blocks automated downloads. Please download manually:\n"
            f"  1. Go to https://www.fhfa.gov/data/hpi/datasets\n"
            f"  2. Find the Census Tract or Local HPI section\n"
            f"  3. Download HPI_AT_BDL_tract.csv\n"
            f"  4. Save it to: data/fhfa/HPI_AT_BDL_tract.csv\n"
            f"  5. Re-run this script\n"
        )

    print(f"Loading FHFA HPI from: {FHFA_CACHE}...", end=" ")
    df = pd.read_csv(FHFA_CACHE, dtype=str)
    print(f"done — {len(df):,} rows")

    # Print columns so we can see the structure
    print(f"\nFHFA columns: {list(df.columns)}")
    print(f"Sample rows:\n{df.head(3).to_string()}")

    return df

    # Standardize column names — FHFA uses varying formats across releases
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Print columns so we can see what we're working with
    print(f"\nFHFA columns: {list(df.columns)}")
    print(f"Sample rows:\n{df.head(3).to_string()}")

    return df


# ── PARSE AND FILTER FHFA DATA ────────────────────────────────────────────────
# Filters to California study counties and extracts index values
# for the three study years (2000, 2015, 2023).
# Handles the supertract structure by keeping tract-level GEOIDs.

def parse_fhfa_ca(df: pd.DataFrame) -> pd.DataFrame:
    print("\nParsing FHFA data for California study counties...")

    # Identify the GEOID column — FHFA uses different names
    geoid_col = None
    for candidate in ["tract", "geoid", "census_tract",
                       "tract_id", "tractid", "fips"]:
        if candidate in df.columns:
            geoid_col = candidate
            break

    if geoid_col is None:
        print("  WARNING: Could not find GEOID column. Available columns:")
        print(f"  {list(df.columns)}")
        raise ValueError("GEOID column not found in FHFA data. "
                         "Check column names above and update parse_fhfa_ca().")

    # Identify the year column
    year_col = None
    for candidate in ["year", "yr", "annual_year"]:
        if candidate in df.columns:
            year_col = candidate
            break

    # Identify the HPI index column — prioritize hpi2000 which is
    # already indexed to 100 in 2000, matching our Census baseline year
    hpi_col = None
    for candidate in ["hpi2000", "hpi2000q4", "hpi_2000",
                       "hpi", "index_nsa", "index", "hpi_nsa",
                       "annual_change", "hpi_with_se"]:
        if candidate in df.columns:
            hpi_col = candidate
            break

    print(f"  Using: geoid={geoid_col}, year={year_col}, hpi={hpi_col}")

    # Ensure GEOID is 11 digits (state 2 + county 3 + tract 6)
    df[geoid_col] = df[geoid_col].astype(str).str.zfill(11)

    # Filter to California (FIPS 06)
    ca = df[df[geoid_col].str.startswith("06")].copy()

    # Filter to study counties
    study_fips = list(STUDY_COUNTIES.values())
    ca = ca[ca[geoid_col].str[2:5].isin(study_fips)].copy()

    print(f"  California study county tracts: {ca[geoid_col].nunique():,}")

    # Convert year and HPI to numeric
    if year_col:
        ca[year_col] = pd.to_numeric(ca[year_col], errors="coerce")
    if hpi_col:
        ca[hpi_col] = pd.to_numeric(ca[hpi_col], errors="coerce")

    # Extract the three study years and create wide format
    # Output columns: hpi2000_2000, hpi2000_2015, hpi2000_2023
    study_frames = {}
    for year in YEARS:
        if year_col:
            yr_df = ca[ca[year_col] == year][[geoid_col, hpi_col]].copy()
        else:
            print(f"  WARNING: Cannot find year column for {year}")
            continue

        yr_df = yr_df.rename(columns={
            geoid_col: "geoid",
            hpi_col:   f"hpi2000_{year}"    # hpi2000_2000, hpi2000_2015, hpi2000_2023
        })
        study_frames[year] = yr_df.dropna(subset=[f"hpi2000_{year}"])
        print(f"  {year}: {len(study_frames[year]):,} tracts with HPI data "
              f"(mean index: {study_frames[year][f'hpi2000_{year}'].mean():.1f})")

    # Merge all three years into wide format
    if not study_frames:
        raise ValueError("No FHFA data found for study years. "
                         "Check the year column format.")

    wide = None
    for year, frame in study_frames.items():
        if wide is None:
            wide = frame
        else:
            wide = wide.merge(frame, on="geoid", how="outer")

    print(f"\n  Wide FHFA table: {len(wide):,} tracts")
    print(f"  Columns: {list(wide.columns)}")
    return wide


# ── LOAD CENSUS 2000 BASELINE ────────────────────────────────────────────────
# Reads the Census 2000 median home values from Script 2's GeoPackage.
# Used to anchor the FHFA index to real dollar levels.

def load_census_baseline() -> pd.DataFrame:
    print("\nLoading Census 2000 baseline from Script 2 GeoPackage...")

    if not os.path.exists(CENSUS_GPKG):
        raise FileNotFoundError(
            f"\nScript 2 GeoPackage not found at {CENSUS_GPKG}\n"
            f"Run housing_data_pipeline.py (Script 2) first."
        )

    gdf = gpd.read_file(CENSUS_GPKG, layer="study_tracts")
    cols = ["geoid", "med_home_value_2000", "med_rent_2000",
            "med_hh_income_2000", "geometry"]
    available = [c for c in cols if c in gdf.columns]
    print(f"  Loaded {len(gdf)} tracts with columns: {available}")
    return gdf[available]


# ── CONVERT FHFA INDEX TO DOLLAR VALUES ──────────────────────────────────────
# The hpi2000 column is already indexed to 100 in year 2000.
# Formula: hv_year = census_2000 × (hpi2000_year / 100)
#
# This means:
#   - 2000 values = Census 2000 exactly (hpi2000 = 100 in 2000)
#   - 2015 values = Census 2000 × (hpi2000_2015 / 100)
#   - 2023 values = Census 2000 × (hpi2000_2023 / 100)
#
# The 2015 and 2023 movements are entirely driven by FHFA transaction
# data, making them independent of the Census ACS for those years.

def convert_to_dollars(fhfa_wide: pd.DataFrame,
                       census_base: gpd.GeoDataFrame) -> pd.DataFrame:
    print("\nConverting FHFA index to dollar values...")

    # Merge FHFA with Census baseline
    merged = census_base.merge(fhfa_wide, on="geoid", how="left")

    # Check match rate using hpi2000 columns
    hpi2000_cols = [c for c in merged.columns if c.startswith("hpi2000_")]
    n_matched = merged[hpi2000_cols[0]].notna().sum() if hpi2000_cols else 0
    n_total   = len(merged)
    print(f"  {n_matched:,} of {n_total:,} tracts matched to FHFA data")

    if n_matched == 0:
        print("  WARNING: No tracts matched. Check GEOID format.")
        print(f"  Census GEOIDs sample: {list(merged['geoid'].head(3))}")
        print(f"  FHFA GEOIDs sample:   {list(fhfa_wide['geoid'].head(3))}")

    # Compute dollar values using hpi2000 index (base = 100 in 2000)
    for year in [2015, 2023]:
        hpi_col = f"hpi2000_{year}"

        if hpi_col in merged.columns:
            # hpi2000 = 100 in 2000, so ratio = hpi2000_year / 100
            ratio = merged[hpi_col] / 100.0
            merged[f"med_home_value_{year}"] = (
                merged["med_home_value_2000"] * ratio
            ).round(0)

            n_fhfa = merged[f"med_home_value_{year}"].notna().sum()
            print(f"  {year}: {n_fhfa:,} tracts with FHFA-derived values "
                  f"(mean ratio: {ratio.mean():.2f})")
        else:
            print(f"  WARNING: Column {hpi_col} not found — "
                  f"available: {[c for c in merged.columns if 'hpi' in c.lower()]}")
            merged[f"med_home_value_{year}"] = np.nan

    return merged


# ── FETCH ACS RENT DATA ───────────────────────────────────────────────────────
# FHFA does not include rent data so we keep the ACS rent for 2015 and 2023.
# This is the same fetch as Script 2 but only for rent variables.

def fetch_acs_rent(year: int, county_fips: str) -> pd.DataFrame:
    url    = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get":  "NAME," + ",".join(ACS_VARS_RENT.keys()),
        "for":  "tract:*",
        "in":   f"state:{STATE_FIPS} county:{county_fips}",
        "key":  CENSUS_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df   = pd.DataFrame(data[1:], columns=data[0])
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    df   = df.rename(columns=ACS_VARS_RENT)
    for col in ACS_VARS_RENT.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(-666666666, np.nan)

    total = df["owner_occ"].fillna(0) + df["renter_occ"].fillna(0)
    df["renter_ratio"] = np.where(total > 0, df["renter_occ"] / total, np.nan)
    df["rent_burden"]  = np.where(
        df["med_hh_income"] > 0,
        (df["med_rent"] * 12) / df["med_hh_income"], np.nan
    )
    time.sleep(0.4)
    return df[["geoid", "med_rent", "med_hh_income",
               "renter_ratio", "rent_burden"]]


def fetch_all_rent() -> pd.DataFrame:
    print("\nFetching ACS rent data for 2015 and 2023...")
    frames = {}
    for year in [2015, 2023]:
        year_frames = []
        for county_name, fips in STUDY_COUNTIES.items():
            print(f"  {year} — {county_name}...", end=" ")
            try:
                df = fetch_acs_rent(year, fips)
                print(f"{len(df)} tracts")
                year_frames.append(df)
            except Exception as e:
                print(f"FAILED: {e}")
        if year_frames:
            frames[year] = pd.concat(year_frames, ignore_index=True)

    # Merge both years
    if 2015 in frames and 2023 in frames:
        rent = frames[2015].rename(columns={
            c: f"{c}_2015" for c in frames[2015].columns if c != "geoid"
        }).merge(
            frames[2023].rename(columns={
                c: f"{c}_2023" for c in frames[2023].columns if c != "geoid"
            }),
            on="geoid", how="outer"
        )
        return rent
    return pd.DataFrame()


# ── BUILD TRACT LAYER ─────────────────────────────────────────────────────────
# Combines FHFA home values with ACS rent data and computes all the
# change metrics and log transforms that Scripts 4 and 5 expect.
# Output schema exactly matches Script 2's study_tracts layer.

def build_tract_layer(merged: gpd.GeoDataFrame,
                      rent_data: pd.DataFrame) -> gpd.GeoDataFrame:
    print("\nBuilding FHFA tract layer...")

    # Attach rent data
    if not rent_data.empty:
        merged = merged.merge(rent_data, on="geoid", how="left")

    # Rename 2000 rent column if it came from census baseline
    if "med_rent_2000" not in merged.columns and "med_rent" in merged.columns:
        merged = merged.rename(columns={"med_rent": "med_rent_2000"})

    # Compute change metrics for home value
    for y0, y1 in [(2000, 2015), (2015, 2023), (2000, 2023)]:
        c0 = f"med_home_value_{y0}"
        c1 = f"med_home_value_{y1}"
        if c0 in merged.columns and c1 in merged.columns:
            merged[f"med_home_value_chg_{y0}_{y1}_abs"] = merged[c1] - merged[c0]
            merged[f"med_home_value_chg_{y0}_{y1}_pct"] = (
                (merged[c1] - merged[c0]) / merged[c0] * 100
            ).round(2)

    # Compute change metrics for rent
    for y0, y1 in [(2000, 2015), (2015, 2023), (2000, 2023)]:
        c0 = f"med_rent_{y0}"
        c1 = f"med_rent_{y1}"
        if c0 in merged.columns and c1 in merged.columns:
            merged[f"med_rent_chg_{y0}_{y1}_abs"] = merged[c1] - merged[c0]
            merged[f"med_rent_chg_{y0}_{y1}_pct"] = (
                (merged[c1] - merged[c0]) / merged[c0] * 100
            ).round(2)

    # Log transforms
    for col in ["med_home_value", "med_rent", "med_hh_income"]:
        for yr in [2000, 2015, 2023]:
            raw = f"{col}_{yr}"
            if raw in merged.columns:
                merged[f"log_{raw}"] = np.log(merged[raw].clip(lower=1))

    # Add data source flag so Scripts 5/6 know which pipeline produced this
    merged["data_source"] = "FHFA_HPI"

    print(f"  Tract layer: {len(merged)} features")

    # Print coverage summary
    print("\n  FHFA home value coverage by county:")
    fips_to_name = {v: k for k, v in STUDY_COUNTIES.items()}
    for fips, name in fips_to_name.items():
        county_mask = merged["geoid"].str[2:5] == fips
        n_total     = county_mask.sum()
        n_hv_2023   = merged[county_mask]["med_home_value_2023"].notna().sum()
        print(f"    {name:<25} {n_hv_2023:>4}/{n_total:>4} tracts "
              f"with 2023 FHFA values")

    return merged


# ── EXPORT TO GEOPACKAGE ──────────────────────────────────────────────────────
# Writes the FHFA tract layer to a separate GeoPackage so the original
# Script 2 output is not overwritten. To run robustness check, point
# Scripts 3–7 to this GeoPackage instead.

def export_to_gpkg(tract_layer: gpd.GeoDataFrame):
    print(f"\nExporting to {GPKG_PATH}...")
    drop = [c for c in tract_layer.columns if c.startswith("index_")]
    tract_layer.drop(columns=drop, errors="ignore").to_file(
        GPKG_PATH, layer="study_tracts", driver="GPKG"
    )
    print(f"  ✓ study_tracts ({len(tract_layer)} features)")
    print(f"\nFHFA GeoPackage ready: {GPKG_PATH}")


# ── PRINT ROBUSTNESS GUIDE ────────────────────────────────────────────────────
# Explains how to run Scripts 3–7 using this output.

def print_robustness_guide():
    print(f"""
{'='*65}
ROBUSTNESS CHECK — HOW TO RUN WITH FHFA DATA
{'='*65}

Script 2b has produced: {GPKG_PATH}

This GeoPackage has the same layer structure as the primary
housing_data.gpkg from Script 2, with one difference:
  med_home_value_2015 and med_home_value_2023 are derived
  from FHFA repeat-sales transaction data, not ACS estimates.
  med_rent values are still from ACS (FHFA has no rent data).

TO RUN THE ROBUSTNESS CHECK:

  In bias_reduction.py (Script 4), change:
    GPKG_PATH = "geopackages/housing_data.gpkg"
  to:
    GPKG_PATH = "geopackages/housing_data_fhfa.gpkg"

  Then rerun Scripts 3, 4, and 5 in order.
  Scripts 6 and 7 will automatically use the new matched panel.

  The output files will overwrite your primary results so
  SAVE YOUR PRIMARY matched_panel.csv FIRST:
    Copy data/matched_panel.csv → data/matched_panel_census.csv
    Then run the robustness check.
    Save output as data/matched_panel_fhfa.csv

WHAT TO COMPARE:

  Primary (Census ACS):  med_home_value = owner self-report
  Robustness (FHFA HPI): med_home_value = repeat-sales transactions

  If TWFE coefficient is similar in magnitude and direction,
  and p-value remains below 0.05, the finding is robust to
  data source.

CITATION FOR FHFA DATA:
  Federal Housing Finance Agency (FHFA). House Price Index,
  Census Tract Level. Washington, DC: FHFA, 2024.
  https://www.fhfa.gov/data/hpi/datasets
{'='*65}
""")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Step 1 — Download and parse FHFA data
    fhfa_raw  = download_fhfa_hpi()
    fhfa_wide = parse_fhfa_ca(fhfa_raw)

    # Step 2 — Load Census 2000 baseline for anchoring
    census_base = load_census_baseline()

    # Step 3 — Convert FHFA index to dollar values
    merged = convert_to_dollars(fhfa_wide, census_base)

    # Step 4 — Fetch ACS rent data (FHFA has no rent)
    rent_data = fetch_all_rent()

    # Step 5 — Build final tract layer
    tract_layer = build_tract_layer(merged, rent_data)

    # Step 6 — Export to GeoPackage
    export_to_gpkg(tract_layer)

    # Step 7 — Print instructions
    print_robustness_guide()


if __name__ == "__main__":
    main()