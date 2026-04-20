# -*- coding: utf-8 -*-
"""
Script 3 — Assign Tract Treatment Groups (Multi-Buffer)
=========================================================
Computes proximity flags and distances for ALL four buffer sizes
in a single pass so Scripts 4 and 5 never need to touch geometry again.

For each buffer radius (2km, 4km, 6km, 8km) and each study period
(2000, 2015, 2023), this script adds a column:

    near_dc_{radius}km_{year}   — True/False proximity flag

The dist_nearest_dc_{year}_km columns are also written so that
Scripts 4 and 5 can reassign treatment groups on the fly by simply
filtering on distance rather than reprocessing geometry.

Must run AFTER:
  Script 1 — OPENING_YEAR_LOOKUP fully filled in
  Script 2 — GeoPackage built with study_tracts layer

Updates the study_tracts layer in the GeoPackage with:
  near_dc_{radius}km_{year}        — proximity flags for all buffers/periods
  dist_nearest_dc_{year}_km        — distance to nearest operational DC
  group_8km                        — default group assignment at 8km
  is_buildout_8km                  — default dummy at 8km
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from opening_year_lookup import (
    load_study_counties,
    build_dated_gdf,
    RAW_DC_CSV,
    OPENING_YEAR_LOOKUP,
    STUDY_COUNTIES,
    EPSG_PROJ,
)

GPKG_PATH   = "geopackages/housing_data.gpkg"
BUFFERS_KM  = [2, 4, 6, 8]   # all buffer radii to compute in one pass
BUFFERS_M   = {b: b * 1000 for b in BUFFERS_KM}
STUDY_YEARS = [2000, 2015, 2023]


# ── MULTI-BUFFER TREATMENT ASSIGNMENT ────────────────────────────────────────
# Replaces the single-buffer assign_tract_treatment from opening_year_lookup.
# Processes each county independently so buffers never cross county lines.
# Computes proximity flags for every combination of buffer × year in one pass.

def assign_tract_treatment_multibuffer(
        tract_gdf: gpd.GeoDataFrame,
        dated_dcs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    tract_gdf = tract_gdf.copy()

    # Ensure CRS matches
    if tract_gdf.crs.to_epsg() != EPSG_PROJ:
        tract_gdf = tract_gdf.to_crs(epsg=EPSG_PROJ)
    if dated_dcs.crs.to_epsg() != EPSG_PROJ:
        dated_dcs = dated_dcs.to_crs(epsg=EPSG_PROJ)

    # Initialize all proximity flag columns to False
    for radius in BUFFERS_KM:
        for year in STUDY_YEARS:
            tract_gdf[f"near_dc_{radius}km_{year}"] = False

    # Initialize distance columns to NaN
    for year in STUDY_YEARS:
        tract_gdf[f"dist_nearest_dc_{year}_km"] = np.nan

    for county in STUDY_COUNTIES:
        county_mask = tract_gdf["county_name"].str.contains(
            county.replace(" County", ""), na=False
        )
        county_dcs = dated_dcs[dated_dcs["county"] == county]

        if len(county_dcs) == 0:
            print(f"  {county}: no DCs found — skipping")
            continue

        # Compute distance from each tract centroid to nearest DC
        # for each study period — do this once per year, reuse for all buffers
        centroids = tract_gdf[county_mask].geometry.centroid

        for year in STUDY_YEARS:
            year_dcs = county_dcs[
                county_dcs[f"operational_{year}"]
            ].copy()

            if len(year_dcs) == 0:
                continue

            # Distance in km from each tract centroid to nearest operational DC
            distances_km = centroids.apply(
                lambda pt: round(
                    year_dcs.geometry.distance(pt).min() / 1000, 3
                )
            )
            tract_gdf.loc[county_mask, f"dist_nearest_dc_{year}_km"] = (
                distances_km.values
            )

            # Set proximity flags for all buffer radii using distance
            for radius in BUFFERS_KM:
                flag_col = f"near_dc_{radius}km_{year}"
                tract_gdf.loc[county_mask, flag_col] = (
                    distances_km <= radius
                ).values

        print(f"  {county}: {county_mask.sum()} tracts processed across "
              f"{len(BUFFERS_KM)} buffer radii")

    # Default group assignment at 8km (primary specification)
    # Scripts 4 and 5 can override this using dist_nearest_dc_2023_km
    tract_gdf["group_8km"]      = "control"
    tract_gdf.loc[
        tract_gdf["near_dc_8km_2023"], "group_8km"
    ] = "buildout"
    tract_gdf["is_buildout_8km"] = (
        tract_gdf["group_8km"] == "buildout"
    ).astype(int)

    # Summary table across all buffer sizes
    print(f"\n{'='*60}")
    print("TREATMENT GROUP SUMMARY BY BUFFER SIZE (2023 operational DCs)")
    print(f"{'='*60}")
    print(f"  {'Buffer':<10} {'Buildout':>10} {'Control':>10} {'Total':>10}")
    print("  " + "─"*40)
    for radius in BUFFERS_KM:
        n_buildout = tract_gdf[f"near_dc_{radius}km_2023"].sum()
        n_control  = (~tract_gdf[f"near_dc_{radius}km_2023"]).sum()
        n_total    = len(tract_gdf)
        print(f"  {radius}km{'':<6} {n_buildout:>10} {n_control:>10} "
              f"{n_total:>10}")

    return tract_gdf


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not OPENING_YEAR_LOOKUP:
        print("ERROR: OPENING_YEAR_LOOKUP is empty.")
        print("Fill in opening years in opening_year_lookup.py first.")
        return

    # Build the dated DC GeoDataFrame from Script 1
    print("Building dated DC dataset from opening year lookup...")
    study     = load_study_counties(RAW_DC_CSV)
    dated_dcs = build_dated_gdf(study)

    # Load the tract layer built by Script 2
    print("\nLoading study_tracts from GeoPackage...")
    tract_gdf = gpd.read_file(GPKG_PATH, layer="study_tracts")
    print(f"  {len(tract_gdf)} tracts loaded")

    # Assign multi-buffer treatment groups
    print("\nAssigning treatment groups across all buffer sizes...")
    tract_gdf = assign_tract_treatment_multibuffer(tract_gdf, dated_dcs)

    # Overwrite the study_tracts layer with all buffer columns
    print("\nSaving updated tract layer to GeoPackage...")
    drop = [c for c in tract_gdf.columns if c.startswith("index_")]
    tract_gdf.drop(columns=drop, errors="ignore").to_file(
        GPKG_PATH, layer="study_tracts", driver="GPKG"
    )
    print(f"  ✓ study_tracts updated in {GPKG_PATH}")

    # Save DC points with dated flags for QGIS reference
    dated_dcs.to_file(GPKG_PATH, layer="dc_points_dated", driver="GPKG")
    print(f"  ✓ dc_points_dated saved")

    # Print column summary so you can verify in QGIS
    new_cols = [c for c in tract_gdf.columns
                if "near_dc" in c or "dist_nearest" in c
                or "group_" in c or "is_buildout" in c]
    print(f"\nNew columns added to study_tracts ({len(new_cols)} total):")
    for col in sorted(new_cols):
        print(f"  {col}")

    print("\nReady to run Script 4 — bias_reduction.py")


if __name__ == "__main__":
    main()