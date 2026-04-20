"""
Script 1 — Opening Year Lookup (Multi-County)
===============================================
Run 1: prints all large DCs across study counties to research.
Run 2: fill in OPENING_YEAR_LOOKUP below, then run again to confirm.

Opening year sources:
  1. Google Earth historical imagery (clock icon — check 2014 vs 2016)
  2. Search "[facility name] [city] opens"
  3. EIA Form 860: eia.gov/electricity/data/eia860
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# ── SETTINGS ──────────────────────────────────────────────────────────────────
RAW_DC_CSV  = "im3_open_source_data_center_atlas_v2026.02.09.csv"
EPSG_PROJ   = 3310     # CA Albers — used for accurate distance calculations
STUDY_START = 2000     # NHGIS Census 2000 SF3a — pre-DC baseline
STUDY_MID   = 2015     # ACS 5-year — during DC buildout
STUDY_END   = 2023     # ACS 5-year — post-buildout
MIN_SQFT    = 100_000  # only facilities this size or larger are included
# Buffer sizes are managed in Script 3 — assign_treatment_groups.py

# Four study counties — Napa excluded from regression (no viable control
# tracts at any buffer size). Santa Clara excluded — DC concentration too
# dominant, tech industry confounding too severe to isolate DC effect.
STUDY_COUNTIES = [
    "Sacramento County",
    "Los Angeles County",
    "Alameda County",
    "San Francisco County",
]

# Napa kept separately for descriptive analysis only
NAPA_COUNTY = "Napa County"

# ── FILL IN AFTER RESEARCH ────────────────────────────────────────────────────
# Add one line per facility: "id_from_printout": opening_year
# IDs are printed when you run this script the first time.

OPENING_YEAR_LOOKUP = {
    # Sacramento
    "00032560292": 1999,   # RagingWire CA1
    "00032560290": 2004,   # RagingWire CA2
    "00190868312": 2013,   # RagingWire CA3

    # Los Angeles
    "00424945845": 2012,   # Serverfarm LAX1
    "00424945852": 2005,   # DTRLAX12 — opened as DC in 2005, built 1978

    # Alameda
    "00384310269": 2006,   # Hurricane Electric Fremont 2 — DC conversion 2006, built 1971

    # Napa
    "00464371693": 1990,   # Kaiser Data Center

    # San Francisco
    "00289620443": 2001,   # Digital Reality SFO10
}

# ── MANUAL ADDITIONS ──────────────────────────────────────────────────────────
# If you find a facility NOT in the IM3 CSV during your research,
# add it here. Use a unique string ID and include opening_year directly.

MANUAL_ADDITIONS = [
    # {
    #     "id":           "MANUAL_001",
    #     "name":         "Example Data Center",
    #     "operator":     "Example Corp",
    #     "county":       "Sacramento County",
    #     "lat":          38.5816,
    #     "lon":         -121.4944,
    #     "sqft":         150000,
    #     "opening_year": 2018,
    # },
]


# ── LOAD & FILTER TO STUDY COUNTIES ───────────────────────────────────────────
# Reads the IM3 CSV, filters to the four study counties, and applies
# the 100k sqft minimum so only large facilities are included.

def load_study_counties(csv_path: str) -> pd.DataFrame:
    df  = pd.read_csv(csv_path, dtype={"id": str})
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")

    mask = (
        df["state_abb"].eq("CA") &
        df["county"].isin(STUDY_COUNTIES) &
        (df["sqft"] >= MIN_SQFT)
    )
    study = df[mask].dropna(subset=["lat", "lon"]).copy()

    if MANUAL_ADDITIONS:
        additions = pd.DataFrame(MANUAL_ADDITIONS)
        additions["state_abb"] = "CA"
        study = pd.concat([study, additions], ignore_index=True)
        print(f"  Appended {len(additions)} manual facility/facilities")

    return study.sort_values(["county", "sqft"], ascending=[True, False])


# ── PRINT FACILITY TABLE ──────────────────────────────────────────────────────
# Groups output by county so you can research one county at a time.
# The ID column is what you use to fill in OPENING_YEAR_LOOKUP above.

def print_facilities(study: pd.DataFrame):
    print(f"\n{'='*85}")
    print(f"LARGE DATA CENTERS (>={MIN_SQFT:,} sqft) — STUDY COUNTIES")
    print(f"{'='*85}")

    for county in STUDY_COUNTIES:
        county_dcs = study[study["county"] == county]
        if len(county_dcs) == 0:
            continue
        print(f"\n{county}  ({len(county_dcs)} facilities)")
        print(f"  {'ID':<15} {'Name':<38} {'Operator':<20} {'Sqft':>12}")
        print("  " + "─"*85)
        for _, r in county_dcs.iterrows():
            sqft = f"{r['sqft']:>12,.0f}" if pd.notna(r["sqft"]) else f"{'unknown':>12}"
            print(f"  {r['id']:<15} {str(r.get('name',''))[:37]:<38} "
                  f"{str(r.get('operator',''))[:19]:<20} {sqft}")

    total      = len(study)
    researched = sum(1 for i in study["id"]
                     if i in OPENING_YEAR_LOOKUP
                     or any(m["id"] == i for m in MANUAL_ADDITIONS))
    print(f"\nTotal: {total}  |  Researched: {researched}  |  "
          f"Remaining: {total - researched}")


# ── BUILD DATED DATASET ───────────────────────────────────────────────────────
# Applies researched opening years and assigns operational flags
# across all three study periods (2000, 2015, 2023).
# Two groups:
#   buildout — DC arrived anywhere in the study window 2000–2023
#   control  — assigned at tract level in Script 3

def build_dated_gdf(study: pd.DataFrame) -> gpd.GeoDataFrame:
    study = study.copy()
    study["opening_year"] = study["id"].map(OPENING_YEAR_LOOKUP)

    # Manual additions carry opening_year directly
    for r in MANUAL_ADDITIONS:
        study.loc[study["id"] == r["id"], "opening_year"] = r.get("opening_year", np.nan)

    # Warn about facilities still missing an opening year
    missing = study[study["opening_year"].isna()]
    if len(missing) > 0:
        print(f"\nWARNING: {len(missing)} facilities still need opening years:")
        for _, r in missing.iterrows():
            print(f"  [{r['county']}]  {r['id']}  {str(r.get('name',''))[:37]}")

    # Keep only verified facilities that opened within the study window
    study = study.dropna(subset=["opening_year"])
    study["opening_year"] = study["opening_year"].astype(int)
    study = study[study["opening_year"] <= STUDY_END]

    # Operational flags across all three periods
    study["operational_2000"] = study["opening_year"] <= STUDY_START
    study["operational_2015"] = study["opening_year"] <= STUDY_MID
    study["operational_2023"] = study["opening_year"] <= STUDY_END

    # Buildout covers the full study window
    study["opened_in_buildout"] = (
        study["opening_year"] <= STUDY_END
    )
    study["est_power_kw"] = study["sqft"].fillna(0) * 0.5

    # Convert to GeoDataFrame projected to CA Albers
    gdf = gpd.GeoDataFrame(
        study,
        geometry=[Point(r.lon, r.lat) for r in study.itertuples()],
        crs="EPSG:4326"
    ).to_crs(epsg=EPSG_PROJ)

    print(f"\n{'='*65}")
    print("VERIFIED FACILITY SUMMARY BY COUNTY")
    print(f"{'='*65}")
    for county in STUDY_COUNTIES:
        sub = gdf[gdf["county"] == county]
        if len(sub) == 0:
            continue
        print(f"\n{county}")
        print(f"  Buildout (any year in study window): {sub['opened_in_buildout'].sum()}")
        for _, r in sub.sort_values("opening_year").iterrows():
            print(f"    {r['opening_year']}  {str(r.get('name',''))[:37]:<38}")

    return gdf


# ── MAIN ──────────────────────────────────────────────────────────────────────
# Treatment group assignment has moved to Script 3 — assign_treatment_groups.py
# which handles all buffer sizes in a single pass.
# First run: prints facility table so you know what to research.
# Second run: builds and displays the cleaned dated dataset.

if __name__ == "__main__":
    study = load_study_counties(RAW_DC_CSV)
    print_facilities(study)

    if OPENING_YEAR_LOOKUP:
        build_dated_gdf(study)
    else:
        print("\nFill in OPENING_YEAR_LOOKUP above, then run again.")