# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:04:04 2026

@author: skell
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# ── SETTINGS ──────────────────────────────────────────────────────────────────
RAW_DC_CSV  = "im3_open_source_data_center_atlas_v2026.02.09.csv"
EPSG_PROJ   = 3310   # CA Albers — used for accurate distance calculations
STUDY_START = 2015
STUDY_END   = 2023
BUFFER_M    = 8000   # 8km proximity threshold for treatment assignment

# ── FILL IN AFTER RESEARCH ────────────────────────────────────────────────────
# Add one line per facility: "id_from_printout": opening_year
# IDs are printed when you run this script the first time.

OPENING_YEAR_LOOKUP = {
    # "00032560292": 1999,  # RagingWire CA1
    # "00032560290": 2004,  # RagingWire CA2
}


# ── LOAD & FILTER TO SACRAMENTO ───────────────────────────────────────────────
# Reads the IM3 CSV and returns only Sacramento County rows,
# sorted largest to smallest so the most impactful facilities appear first.

def load_sacramento(csv_path: str) -> pd.DataFrame:
    df  = pd.read_csv(csv_path, dtype={"id": str})
    sac = df[df["county"].str.contains("Sacramento", na=False)].copy()
    sac = sac.dropna(subset=["lat", "lon"])
    sac["sqft"] = pd.to_numeric(sac["sqft"], errors="coerce")
    return sac.sort_values("sqft", ascending=False)


# ── PRINT FACILITY TABLE ──────────────────────────────────────────────────────
# Shows every Sacramento facility with its ID, name, operator, and size.
# The ID column is what you use to fill in OPENING_YEAR_LOOKUP above.

def print_facilities(sac: pd.DataFrame):
    print(f"\n{'ID':<15} {'Name':<40} {'Operator':<20} {'Sqft':>12}")
    print("─" * 88)
    for _, r in sac.iterrows():
        sqft = f"{r['sqft']:>12,.0f}" if pd.notna(r["sqft"]) else f"{'unknown':>12}"
        print(f"{r['id']:<15} {str(r.get('name',''))[:39]:<40} "
              f"{str(r.get('operator',''))[:19]:<20} {sqft}")
    print(f"\nTotal: {len(sac)}  |  Researched: {len(OPENING_YEAR_LOOKUP)}"
          f"  |  Remaining: {len(sac) - len(OPENING_YEAR_LOOKUP)}")


# ── BUILD DATED DATASET ───────────────────────────────────────────────────────
# Applies your researched opening years to each facility.
# Assigns three flags used downstream by the regression scripts:
#   operational_2015 — was this DC open by 2015? (long-exposed group)
#   operational_2023 — was this DC open by 2023? (needed for both groups)
#   opened_in_window — did this DC open between 2016–2023? (treated group)
# Facilities with unknown opening years or that opened after 2023 are dropped.

def build_dated_sacramento_gdf(sac: pd.DataFrame) -> gpd.GeoDataFrame:
    sac = sac.copy()
    sac["opening_year"] = sac["id"].map(OPENING_YEAR_LOOKUP)

    # Warn about any facilities still missing an opening year
    missing = sac[sac["opening_year"].isna()]
    if len(missing) > 0:
        print(f"\nWARNING: {len(missing)} facilities still need opening years:")
        for _, r in missing.iterrows():
            print(f"  {r['id']}  {str(r.get('name',''))[:39]}")

    # Keep only verified facilities that opened within the study window
    sac = sac.dropna(subset=["opening_year"])
    sac["opening_year"] = sac["opening_year"].astype(int)
    sac = sac[sac["opening_year"] <= STUDY_END]

    # Assign status flags for treatment group assignment in Script 3
    sac["operational_2015"] = sac["opening_year"] <= STUDY_START
    sac["operational_2023"] = sac["opening_year"] <= STUDY_END
    sac["opened_in_window"] = (
        (sac["opening_year"] > STUDY_START) &
        (sac["opening_year"] <= STUDY_END)
    )
    sac["est_power_kw"] = sac["sqft"].fillna(0) * 0.5

    # Convert to GeoDataFrame projected to CA Albers for accurate distances
    gdf = gpd.GeoDataFrame(
        sac,
        geometry=[Point(r.lon, r.lat) for r in sac.itertuples()],
        crs="EPSG:4326"
    ).to_crs(epsg=EPSG_PROJ)

    # Print summary so you can verify assignments look correct
    print(f"\nVerified facilities: {len(gdf)}")
    print(f"  Long-exposed (open by {STUDY_START}): {gdf['operational_2015'].sum()}")
    print(f"  Treated (opened {STUDY_START+1}–{STUDY_END}): {gdf['opened_in_window'].sum()}")
    print(f"\n  {'Year':<6} {'Name':<40} {'Group'}")
    print("  " + "─" * 60)
    for _, r in gdf.sort_values("opening_year").iterrows():
        group = "long-exposed" if r["operational_2015"] else "treated"
        print(f"  {r['opening_year']:<6} {str(r.get('name',''))[:39]:<40} {group}")

    return gdf


# ── ASSIGN TRACT TREATMENT GROUPS ─────────────────────────────────────────────
# Called by Script 3 after Script 2 has built the tract GeoDataFrame.
# Buffers each year's operational DCs and checks which tracts fall inside.
# Adds near_dc_2015, near_dc_2023, distance columns, and the group label
# (control / treated / long_exposed) used by the regression in Script 5.

def assign_tract_treatment(tract_gdf: gpd.GeoDataFrame,
                            dated_dcs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tract_gdf = tract_gdf.copy()

    for year, col in [(2015, "near_dc_2015"), (2023, "near_dc_2023")]:
        year_dcs = dated_dcs[dated_dcs[f"operational_{year}"]].copy()
        if len(year_dcs) == 0:
            tract_gdf[col] = False
            tract_gdf[f"dist_nearest_dc_{year}_km"] = np.nan
            continue

        # Buffer all DCs for that year and flag tracts that intersect
        union = year_dcs.geometry.buffer(BUFFER_M).union_all()
        tract_gdf[col] = tract_gdf.geometry.intersects(union)

        # Distance from each tract centroid to the nearest DC that year
        centroids = tract_gdf.geometry.centroid
        tract_gdf[f"dist_nearest_dc_{year}_km"] = centroids.apply(
            lambda pt: round(year_dcs.geometry.distance(pt).min() / 1000, 2)
        )
        print(f"  {year}: {tract_gdf[col].sum()} tracts within "
              f"{BUFFER_M/1000:.0f}km of a DC")

    # Assign each tract to exactly one group based on its proximity history
    tract_gdf["group"] = "control"
    tract_gdf.loc[(~tract_gdf["near_dc_2015"]) & tract_gdf["near_dc_2023"],
                  "group"] = "treated"
    tract_gdf.loc[tract_gdf["near_dc_2015"] & tract_gdf["near_dc_2023"],
                  "group"] = "long_exposed"

    # Binary dummies used as regression variables in Script 5
    tract_gdf["is_treated"]      = (tract_gdf["group"] == "treated").astype(int)
    tract_gdf["is_long_exposed"] = (tract_gdf["group"] == "long_exposed").astype(int)

    print(f"  Control: {(tract_gdf['group']=='control').sum()}  "
          f"Treated: {(tract_gdf['group']=='treated').sum()}  "
          f"Long-exposed: {(tract_gdf['group']=='long_exposed').sum()}")

    return tract_gdf


# ── MAIN ──────────────────────────────────────────────────────────────────────
# First run: loads CSV and prints the facility table.
# Second run: also builds and displays the cleaned dated dataset.

if __name__ == "__main__":
    sac = load_sacramento(RAW_DC_CSV)
    print_facilities(sac)

    if OPENING_YEAR_LOOKUP:
        build_dated_sacramento_gdf(sac)
    else:
        print("\nFill in OPENING_YEAR_LOOKUP above, then run again.")