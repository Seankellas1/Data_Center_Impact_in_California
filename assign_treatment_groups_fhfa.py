"""
Script 3b — Assign Tract Treatment Groups (FHFA Robustness Check)
==================================================================
Identical to Script 3 but reads from and writes to the FHFA
GeoPackage produced by Script 2b. The primary Census GeoPackage
is never touched.

Run after Script 2b — housing_data_pipeline_fhfa.py
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

# ── FHFA-specific paths — primary Census paths never touched ──────────────────
GPKG_PATH   = "geopackages/housing_data_fhfa.gpkg"
BUFFERS_KM  = [2, 4, 6, 8]
BUFFERS_M   = {b: b * 1000 for b in BUFFERS_KM}
STUDY_YEARS = [2000, 2015, 2023]


def assign_tract_treatment_multibuffer(
        tract_gdf: gpd.GeoDataFrame,
        dated_dcs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    tract_gdf = tract_gdf.copy()

    if tract_gdf.crs.to_epsg() != EPSG_PROJ:
        tract_gdf = tract_gdf.to_crs(epsg=EPSG_PROJ)
    if dated_dcs.crs.to_epsg() != EPSG_PROJ:
        dated_dcs = dated_dcs.to_crs(epsg=EPSG_PROJ)

    for radius in BUFFERS_KM:
        for year in STUDY_YEARS:
            tract_gdf[f"near_dc_{radius}km_{year}"] = False

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

        centroids = tract_gdf[county_mask].geometry.centroid

        for year in STUDY_YEARS:
            year_dcs = county_dcs[
                county_dcs[f"operational_{year}"]
            ].copy()

            if len(year_dcs) == 0:
                continue

            distances_km = centroids.apply(
                lambda pt: round(
                    year_dcs.geometry.distance(pt).min() / 1000, 3
                )
            )
            tract_gdf.loc[county_mask, f"dist_nearest_dc_{year}_km"] = (
                distances_km.values
            )

            for radius in BUFFERS_KM:
                flag_col = f"near_dc_{radius}km_{year}"
                tract_gdf.loc[county_mask, flag_col] = (
                    distances_km <= radius
                ).values

        print(f"  {county}: {county_mask.sum()} tracts processed")

    tract_gdf["group_8km"]       = "control"
    tract_gdf.loc[
        tract_gdf["near_dc_8km_2023"], "group_8km"
    ] = "buildout"
    tract_gdf["is_buildout_8km"] = (
        tract_gdf["group_8km"] == "buildout"
    ).astype(int)

    print(f"\nTreatment group summary (FHFA sample):")
    print(f"  Control:  {(tract_gdf['group_8km']=='control').sum()} tracts")
    print(f"  Buildout: {(tract_gdf['group_8km']=='buildout').sum()} tracts")

    return tract_gdf


def main():
    if not OPENING_YEAR_LOOKUP:
        print("ERROR: OPENING_YEAR_LOOKUP is empty in opening_year_lookup.py")
        return

    print("Script 3b — FHFA Robustness Check")
    print(f"Reading from: {GPKG_PATH}")

    study     = load_study_counties(RAW_DC_CSV)
    dated_dcs = build_dated_gdf(study)

    print(f"\nLoading FHFA study_tracts from GeoPackage...")
    tract_gdf = gpd.read_file(GPKG_PATH, layer="study_tracts")
    print(f"  {len(tract_gdf)} tracts loaded")
    print(f"  Columns: {list(tract_gdf.columns)}")

    # Derive county_name from geoid if not present
    # GEOID format: SSCCCTTTTTT (state 2 + county 3 + tract 6)
    if "county_name" not in tract_gdf.columns:
        print("  county_name not found — deriving from geoid...")
        FIPS_TO_COUNTY = {
            "06067": "Sacramento County",
            "06037": "Los Angeles County",
            "06001": "Alameda County",
            "06075": "San Francisco County",
            "06055": "Napa County",
        }
        tract_gdf["county_fips"] = tract_gdf["geoid"].astype(str).str[:5]
        tract_gdf["county_name"] = tract_gdf["county_fips"].map(FIPS_TO_COUNTY)
        print(f"  county_name derived: "
              f"{tract_gdf['county_name'].notna().sum()} tracts matched")

    print("\nAssigning treatment groups...")
    tract_gdf = assign_tract_treatment_multibuffer(tract_gdf, dated_dcs)

    print("\nSaving updated tract layer to FHFA GeoPackage...")
    drop = [c for c in tract_gdf.columns if c.startswith("index_")]
    tract_gdf.drop(columns=drop, errors="ignore").to_file(
        GPKG_PATH, layer="study_tracts", driver="GPKG"
    )
    print(f"  ✓ study_tracts updated in {GPKG_PATH}")

    dated_dcs.to_file(GPKG_PATH, layer="dc_points_dated", driver="GPKG")
    print(f"  ✓ dc_points_dated saved")

    print("\nReady to run Script 4b — bias_reduction_fhfa.py")


if __name__ == "__main__":
    main()