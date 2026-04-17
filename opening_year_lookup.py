# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:04:04 2026

@author: skell
"""
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
STUDY_START = 2000   # Census 2000 Decennial — pre-DC baseline
STUDY_MID   = 2015   # ACS 5-year — during DC buildout
STUDY_END   = 2023   # ACS 5-year — post-buildout
BUFFER_M    = 8000     # 8km proximity threshold for treatment assignment
MIN_SQFT    = 100_000  # only facilities this size or larger are included

# Five study counties — Sacramento kept as anchor, others added for
# generalizability. Napa included as a contrast case (non-tech context).
# Santa Clara excluded — DC concentration too dominant, tech industry
# confounding too severe to isolate DC effect cleanly.
STUDY_COUNTIES = [
    "Sacramento County",
    "Los Angeles County",
    "Alameda County",
    "Napa County",
    "San Francisco County",
]

# ── FILL IN AFTER RESEARCH ────────────────────────────────────────────────────
# Add one line per facility: "id_from_printout": opening_year
# IDs are printed when you run this script the first time.

OPENING_YEAR_LOOKUP = {
    # Sacramento
     "00032560292": 1999,  # RagingWire CA1
     "00032560290": 2004,  # RagingWire CA2
     "00190868312": 2013,  # RagingWire CA3

    # Los Angeles
     "00424945845":  2012, # Serverfarm-LAX1
     "00424945852":  2005, # DTRLAX12 #DTRLAX12 opened as a DC in 2005, 
                                       # it was built in 1978 though. 

    # Alameda
     "00384310269":  2006, #HurricaneElectricFremont2 
# Similar to DTRLAX12, HurricaneElectricFremont2 started as a regular building 
# in 1971, but transitioned to a DC in 2006

    # Napa
     "00464371693":  1990, #KaiserDataCenter

    # San Francisco
    "00289620443":  2001, #Digital RealitySFO10
}

# ── MANUAL ADDITIONS ──────────────────────────────────────────────────────────
# If you find a facility NOT in the IM3 CSV during your research,
# add it here. Use a unique string ID and include opening_year directly.
# These are appended to the dataset before any analysis runs.

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
# Reads the IM3 CSV, filters to the five study counties, and applies
# the 100k sqft minimum so only large facilities are included.
# Manual additions are appended before returning.

def load_study_counties(csv_path: str) -> pd.DataFrame:
    df  = pd.read_csv(csv_path, dtype={"id": str})
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")

    # Filter to study counties and minimum size threshold
    mask = (
        df["state_abb"].eq("CA") &
        df["county"].isin(STUDY_COUNTIES) &
        (df["sqft"] >= MIN_SQFT)
    )
    study = df[mask].dropna(subset=["lat", "lon"]).copy()

    # Append manually added facilities
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
    print(f"LARGE DATA CENTERS (≥{MIN_SQFT:,} sqft) — STUDY COUNTIES")
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

    total     = len(study)
    researched = sum(1 for i in study["id"] if i in OPENING_YEAR_LOOKUP
                     or any(m["id"] == i for m in MANUAL_ADDITIONS))
    print(f"\nTotal facilities: {total}  |  "
          f"Researched: {researched}  |  "
          f"Remaining: {total - researched}")


# ── BUILD DATED DATASET ───────────────────────────────────────────────────────
# Applies researched opening years to each facility.
# Assigns operational flags used by Scripts 2–5 for treatment assignment.
# Facilities with unknown opening years or opened after 2023 are dropped.

def build_dated_gdf(study: pd.DataFrame) -> gpd.GeoDataFrame:
    study = study.copy()
    study["opening_year"] = study["id"].map(OPENING_YEAR_LOOKUP)

    # Manual additions carry opening_year directly — fill those in
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

    # Assign status flags for treatment group assignment in Script 3
    study["operational_2015"] = study["opening_year"] <= STUDY_START
    study["operational_2023"] = study["opening_year"] <= STUDY_END
    study["opened_in_window"] = (
        (study["opening_year"] > STUDY_START) &
        (study["opening_year"] <= STUDY_END)
    )
    study["est_power_kw"] = study["sqft"].fillna(0) * 0.5

    # Convert to GeoDataFrame projected to CA Albers for accurate distances
    gdf = gpd.GeoDataFrame(
        study,
        geometry=[Point(r.lon, r.lat) for r in study.itertuples()],
        crs="EPSG:4326"
    ).to_crs(epsg=EPSG_PROJ)

    # Print summary grouped by county so you can verify assignments
    print(f"\n{'='*65}")
    print("VERIFIED FACILITY SUMMARY BY COUNTY")
    print(f"{'='*65}")
    for county in STUDY_COUNTIES:
        sub = gdf[gdf["county"] == county]
        if len(sub) == 0:
            continue
        print(f"\n{county}")
        print(f"  Long-exposed (open by {STUDY_START}): {sub['operational_2015'].sum()}")
        print(f"  Treated (opened {STUDY_START+1}–{STUDY_END}):  {sub['opened_in_window'].sum()}")
        for _, r in sub.sort_values("opening_year").iterrows():
            group = "long-exposed" if r["operational_2015"] else "treated"
            print(f"    {r['opening_year']}  {str(r.get('name',''))[:37]:<38} [{group}]")

    return gdf


# ── ASSIGN TRACT TREATMENT GROUPS ─────────────────────────────────────────────
# Called by Script 3 after Script 2 has built the tract GeoDataFrame.
# Operates on each county separately so a DC in LA doesn't flag
# tracts in Sacramento as treated.
# Adds near_dc_2015, near_dc_2023, distance columns, group label,
# and binary dummies used as regression variables in Script 5.

def assign_tract_treatment(tract_gdf: gpd.GeoDataFrame,
                            dated_dcs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tract_gdf = tract_gdf.copy()

    # Initialize columns
    tract_gdf["near_dc_2015"] = False
    tract_gdf["near_dc_2023"] = False
    tract_gdf["dist_nearest_dc_2015_km"] = np.nan
    tract_gdf["dist_nearest_dc_2023_km"] = np.nan

    # Process each county independently — critical so LA DCs don't
    # accidentally flag Sacramento tracts as treated
    for county in STUDY_COUNTIES:
        county_tracts = tract_gdf["county_name"].str.contains(
            county.replace(" County", ""), na=False
        )
        county_dcs = dated_dcs[dated_dcs["county"] == county]

        for year, col in [(2015, "near_dc_2015"), (2023, "near_dc_2023")]:
            year_dcs = county_dcs[county_dcs[f"operational_{year}"]].copy()
            if len(year_dcs) == 0:
                continue

            # Buffer all DCs for that year and flag tracts that intersect
            union = year_dcs.geometry.buffer(BUFFER_M).union_all()
            tract_gdf.loc[county_tracts, col] = (
                tract_gdf[county_tracts].geometry.intersects(union)
            )

            # Distance from each tract centroid to nearest DC that year
            centroids = tract_gdf[county_tracts].geometry.centroid
            tract_gdf.loc[county_tracts, f"dist_nearest_dc_{year}_km"] = (
                centroids.apply(
                    lambda pt: round(
                        year_dcs.geometry.distance(pt).min() / 1000, 2
                    )
                ).values
            )

        n_treated = tract_gdf[county_tracts & tract_gdf["near_dc_2023"]].shape[0]
        print(f"  {county}: {n_treated} tracts within {BUFFER_M/1000:.0f}km of a DC")

    # Assign each tract to exactly one group based on proximity history
    tract_gdf["group"] = "control"
    tract_gdf.loc[
        (~tract_gdf["near_dc_2015"]) & tract_gdf["near_dc_2023"],
        "group"
    ] = "treated"
    tract_gdf.loc[
        tract_gdf["near_dc_2015"] & tract_gdf["near_dc_2023"],
        "group"
    ] = "long_exposed"

    # Binary dummies used as regression variables in Script 5
    tract_gdf["is_treated"]      = (tract_gdf["group"] == "treated").astype(int)
    tract_gdf["is_long_exposed"] = (tract_gdf["group"] == "long_exposed").astype(int)

    print(f"\n  Total across all counties:")
    print(f"    Control:      {(tract_gdf['group']=='control').sum()} tracts")
    print(f"    Treated:      {(tract_gdf['group']=='treated').sum()} tracts")
    print(f"    Long-exposed: {(tract_gdf['group']=='long_exposed').sum()} tracts")

    return tract_gdf


# ── MAIN ──────────────────────────────────────────────────────────────────────
# First run: loads CSV and prints the facility table by county.
# Second run: also builds and displays the cleaned dated dataset.

if __name__ == "__main__":
    study = load_study_counties(RAW_DC_CSV)
    print_facilities(study)

    if OPENING_YEAR_LOOKUP:
        build_dated_gdf(study)
    else:
        print("\nFill in OPENING_YEAR_LOOKUP above, then run again.")