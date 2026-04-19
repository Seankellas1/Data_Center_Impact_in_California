# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:24:06 2026

@author: skell
"""

"""
Script 3 — Assign Tract Treatment Groups
==========================================
Applies dated DC opening years to assign each census tract
to one of three treatment groups:

  control:  no DC nearby in any period
  pre2000:  DC present before 2000 (Napa Kaiser Data Center)
  buildout: DC arrived between 2000 and 2023

Must run AFTER:
  Script 1 — OPENING_YEAR_LOOKUP fully filled in
  Script 2 — GeoPackage built with study_tracts layer

Updates the study_tracts layer in the GeoPackage with:
  near_dc_2000 / near_dc_2015 / near_dc_2023  — proximity flags
  dist_nearest_dc_{year}_km                    — distance columns
  group        — control / pre2000 / buildout
  is_pre2000, is_buildout                      — regression dummies
"""

import geopandas as gpd
from opening_year_lookup import (
    load_study_counties,
    build_dated_gdf,
    assign_tract_treatment,
    RAW_DC_CSV,
    OPENING_YEAR_LOOKUP,
)

GPKG_PATH = "geopackages/housing_data.gpkg"


def main():
    # Verify opening years are filled in before proceeding
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

    # Assign treatment groups based on DC proximity across all three periods
    print("\nAssigning treatment groups...")
    tract_gdf = assign_tract_treatment(tract_gdf, dated_dcs)

    # Overwrite the study_tracts layer with updated treatment columns
    print("\nSaving updated tract layer to GeoPackage...")
    drop = [c for c in tract_gdf.columns if c.startswith("index_")]
    tract_gdf.drop(columns=drop, errors="ignore").to_file(
        GPKG_PATH, layer="study_tracts", driver="GPKG"
    )
    print(f"  ✓ study_tracts updated in {GPKG_PATH}")

    # Also save DC points with dated flags for QGIS reference
    dated_dcs.to_file(GPKG_PATH, layer="dc_points_dated", driver="GPKG")
    print(f"  ✓ dc_points_dated saved")

    print("\nReady to run Script 4 — bias_reduction.py")


if __name__ == "__main__":
    main()