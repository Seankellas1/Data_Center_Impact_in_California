# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:03:07 2026

@author: skell
"""

"""
Script 2 — Housing Data Pipeline (Multi-County)
=================================================
Fetches housing data across three time periods and five study counties.

Time periods:
  2000 — NHGIS Census SF3a tract-level CSV (pre-DC baseline)
  2015 — ACS 5-year via Census API (during DC buildout)
  2023 — ACS 5-year via Census API (post-buildout / hyperscale era)

NHGIS variable codes (from codebook):
  GCL001 — Median value of owner-occupied housing units
  GBO001 — Median gross rent
  GMY001 — Median household income in 1999 dollars

Place your NHGIS CSV at: data/nhgis/nhgis0001_ds151_2000_tract.csv

Study counties: Sacramento, Los Angeles, Alameda, Napa, San Francisco
Output: geopackages/housing_data.gpkg

Requires: export CENSUS_API_KEY="your_key"
Install:  pip install pandas geopandas shapely requests tqdm
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ── SETTINGS ──────────────────────────────────────────────────────────────────
RAW_DC_CSV     = "im3_open_source_data_center_atlas_v2026.02.09.csv"
OUTPUT_DIR     = "geopackages"
GPKG_PATH      = os.path.join(OUTPUT_DIR, "housing_data.gpkg")
NHGIS_CSV      = "data/nhgis/nhgis0001_ds151_2000_tract.csv"
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "314253ac718eac9ce5891234409bba3d70aa48d6")
EPSG_GEO       = 4326
EPSG_PROJ      = 3310
STATE_FIPS     = "06"
YEARS          = [2000, 2015, 2023]
ACS_YEARS      = [2015, 2023]
MIN_SQFT       = 100_000
BUFFER_M       = 8000

# Study counties with Census FIPS codes
STUDY_COUNTIES = {
    "Sacramento County":    "067",
    "Los Angeles County":   "037",
    "Alameda County":       "001",
    "Napa County":          "055",
    "San Francisco County": "075",
}

# ACS variable codes — used for 2015 and 2023 Census API pulls
ACS_VARS = {
    "B25077_001E": "med_home_value",
    "B25064_001E": "med_rent",
    "B19013_001E": "med_hh_income",
    "B25003_002E": "owner_occ",
    "B25003_003E": "renter_occ",
}

# NHGIS 2000 column mapping — from codebook
# GMY001 = median household income in 1999 dollars (note: prior year income)
NHGIS_VARS = {
    "GCL001": "med_home_value",
    "GBO001": "med_rent",
    "GMY001": "med_hh_income",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/nhgis", exist_ok=True)


# ── LOAD STUDY DATA CENTERS ───────────────────────────────────────────────────
# Filters IM3 CSV to study counties and 100k+ sqft facilities only.

def load_study_dcs() -> gpd.GeoDataFrame:
    df = pd.read_csv(RAW_DC_CSV, dtype={"id": str})
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")

    mask = (
        df["state_abb"].eq("CA") &
        df["county"].isin(STUDY_COUNTIES.keys()) &
        (df["sqft"] >= MIN_SQFT)
    )
    ca = df[mask].dropna(subset=["lat", "lon"]).copy()

    bins   = [0, 100_000, 200_000, 500_000, float("inf")]
    labels = ["small", "medium", "large", "hyperscale"]
    ca["scale_class"]  = pd.cut(ca["sqft"], bins=bins, labels=labels).astype(str)
    ca["est_power_kw"] = ca["sqft"].fillna(0) * 0.5

    gdf = gpd.GeoDataFrame(
        ca,
        geometry=[Point(r.lon, r.lat) for r in ca.itertuples()],
        crs=f"EPSG:{EPSG_GEO}"
    ).to_crs(epsg=EPSG_PROJ)

    print(f"Loaded {len(gdf)} large DCs across study counties")
    print(gdf.groupby("county")["id"].count().to_string())
    return gdf


# ── LOAD NHGIS 2000 DATA ──────────────────────────────────────────────────────
# Reads the NHGIS Census 2000 SF3a tract-level CSV.
# GISJOIN format: G + state(2) + 0 + county(3) + 0 + tract(6)
# Standard 11-digit GEOID = STATEA + COUNTYA + TRACTA

def load_nhgis_2000() -> pd.DataFrame:
    if not os.path.exists(NHGIS_CSV):
        raise FileNotFoundError(
            f"\nNHGIS CSV not found at {NHGIS_CSV}\n"
            f"Download from nhgis.org and place at that path."
        )

    print("Loading NHGIS 2000 SF3a data...", end=" ")
    df = pd.read_csv(NHGIS_CSV, dtype=str, encoding="latin-1")

    # Build standard 11-digit GEOID from NHGIS context fields
    df["geoid"] = df["STATEA"] + df["COUNTYA"] + df["TRACTA"]

    # Filter to study counties
    study_fips = list(STUDY_COUNTIES.values())
    df = df[
        (df["STATEA"] == "06") &
        (df["COUNTYA"].isin(study_fips))
    ].copy()

    # Rename NHGIS variable codes to readable names
    df = df.rename(columns=NHGIS_VARS)

    # Convert to numeric — NHGIS uses 0 for suppressed values
    for col in NHGIS_VARS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace(0, np.nan)

    # Renter ratio not available in this extract — set NaN
    df["renter_ratio"] = np.nan
    df["rent_burden"]  = np.where(
        df["med_hh_income"].fillna(0) > 0,
        (df["med_rent"] * 12) / df["med_hh_income"],
        np.nan
    )

    out = df[["geoid", "med_home_value", "med_rent",
              "med_hh_income", "renter_ratio", "rent_burden"]]
    print(f"{len(out)} tracts across study counties")
    return out


# ── FETCH ACS DATA ────────────────────────────────────────────────────────────
# Pulls ACS 5-year estimates for 2015 and 2023 for each study county.

def fetch_acs_county(year: int, county_fips: str) -> pd.DataFrame:
    url    = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get":  "NAME," + ",".join(ACS_VARS.keys()),
        "for":  "tract:*",
        "in":   f"state:{STATE_FIPS} county:{county_fips}",
        "key":  CENSUS_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df   = pd.DataFrame(data[1:], columns=data[0])
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    df = df.rename(columns=ACS_VARS)
    for col in ACS_VARS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(-666666666, np.nan)

    total = df["owner_occ"].fillna(0) + df["renter_occ"].fillna(0)
    df["renter_ratio"] = np.where(total > 0, df["renter_occ"] / total, np.nan)
    df["rent_burden"]  = np.where(
        df["med_hh_income"] > 0,
        (df["med_rent"] * 12) / df["med_hh_income"], np.nan
    )
    time.sleep(0.4)
    return df[["geoid", "med_home_value", "med_rent", "med_hh_income",
               "renter_ratio", "rent_burden"]]


# ── FETCH ALL DATA ─────────────────────────────────────────────────────────────
# Loads 2000 from NHGIS and fetches 2015/2023 from Census API.
# Merges all three into a wide-format DataFrame keyed on geoid.

def fetch_all_data() -> pd.DataFrame:
    if CENSUS_API_KEY in ("YOUR_KEY_HERE", "", None):
        raise EnvironmentError(
            "\nCensus API key not set.\n"
            "Get a free key at: api.census.gov/data/key_signup.html\n"
            "Set it before running:\n"
            "  Windows:  set CENSUS_API_KEY=your_key\n"
            "  Mac/Linux: export CENSUS_API_KEY=your_key"
        )

    # 2000 baseline from NHGIS CSV
    print("\nLoading 2000 baseline from NHGIS...")
    nhgis_2000 = load_nhgis_2000()

    # 2015 and 2023 from Census ACS API
    print("\nFetching ACS 5-year data (2015 and 2023)...")
    frames = {}
    for year in ACS_YEARS:
        year_frames = []
        for county_name, fips in STUDY_COUNTIES.items():
            print(f"  {year} — {county_name}...", end=" ")
            try:
                df = fetch_acs_county(year, fips)
                if df.empty or "geoid" not in df.columns:
                    print("SKIPPED (empty response)")
                else:
                    print(f"{len(df)} tracts")
                    year_frames.append(df)
            except Exception as e:
                print(f"FAILED: {e}")

        if not year_frames:
            raise RuntimeError(
                f"\nAll ACS {year} fetches failed.\n"
                f"Check CENSUS_API_KEY and internet connection."
            )
        frames[year] = pd.concat(year_frames, ignore_index=True)

    # Merge all three periods into wide format
    wide = (
        nhgis_2000.rename(columns={c: f"{c}_2000"
                                    for c in nhgis_2000.columns if c != "geoid"})
        .merge(
            frames[2015].rename(columns={c: f"{c}_2015"
                                          for c in frames[2015].columns if c != "geoid"}),
            on="geoid", how="outer"
        )
        .merge(
            frames[2023].rename(columns={c: f"{c}_2023"
                                          for c in frames[2023].columns if c != "geoid"}),
            on="geoid", how="outer"
        )
    )

    # Compute change metrics for all three period pairs
    for col in ["med_home_value", "med_rent", "med_hh_income"]:
        for y0, y1 in [(2000, 2015), (2015, 2023), (2000, 2023)]:
            c0, c1 = f"{col}_{y0}", f"{col}_{y1}"
            if c0 in wide.columns and c1 in wide.columns:
                wide[f"{col}_chg_{y0}_{y1}_abs"] = wide[c1] - wide[c0]
                wide[f"{col}_chg_{y0}_{y1}_pct"] = (
                    (wide[c1] - wide[c0]) / wide[c0] * 100
                ).round(2)

    print(f"\nFinal wide table: {len(wide)} tracts, {len(wide.columns)} columns")
    print("Note: outer merge — some 2000 tracts may not match 2020 boundaries")
    return wide


# ── LOAD TRACT GEOMETRIES ─────────────────────────────────────────────────────
# Downloads CA tract boundaries and filters to study counties.
# Cached after first download so subsequent runs are instant.

def load_tract_geometries() -> gpd.GeoDataFrame:
    cache = f"data/tl_2020_{STATE_FIPS}_tract.zip"
    if not os.path.exists(cache):
        url = (f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/"
               f"tl_2020_{STATE_FIPS}_tract.zip")
        print("Downloading CA tract boundaries (~30 MB)...", end=" ")
        r = requests.get(url, stream=True, timeout=120)
        with open(cache, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print("done")

    tracts = gpd.read_file(f"zip://{cache}").to_crs(epsg=EPSG_PROJ)
    tracts["geoid"] = tracts["STATEFP"] + tracts["COUNTYFP"] + tracts["TRACTCE"]

    study_fips   = list(STUDY_COUNTIES.values())
    tracts       = tracts[tracts["COUNTYFP"].isin(study_fips)].copy()
    fips_to_name = {v: k for k, v in STUDY_COUNTIES.items()}
    tracts["county_name"] = tracts["COUNTYFP"].map(fips_to_name)

    print(f"Loaded {len(tracts)} tracts across study counties")
    return tracts[["geoid", "county_name", "COUNTYFP", "geometry"]]


# ── BUILD TRACT LAYER ─────────────────────────────────────────────────────────
# Joins all Census data to tract geometries.
# This is the primary analysis layer used by Scripts 3, 4, and 5.

def build_tract_layer(tract_geom: gpd.GeoDataFrame,
                      census_data: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = tract_geom.merge(census_data, on="geoid", how="left")

    # Log transforms for regression
    for col in ["med_home_value", "med_rent", "med_hh_income"]:
        for yr in YEARS:
            raw = f"{col}_{yr}"
            if raw in gdf.columns:
                gdf[f"log_{raw}"] = np.log(gdf[raw].clip(lower=1))

    print(f"Tract layer: {len(gdf)} features")
    return gdf


# ── BUILD DC BUFFER LAYERS ────────────────────────────────────────────────────
# Creates concentric impact rings around each study DC for QGIS overlay.

def build_dc_buffers(dcs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rings = []
    for _, dc in dcs.iterrows():
        prev = None
        for radius, label in [(2000, "0-2km"), (5000, "2-5km"), (8000, "5-8km")]:
            circle  = dc.geometry.buffer(radius)
            annulus = circle.difference(prev) if prev else circle
            prev    = circle
            rings.append({
                "dc_id":      dc["id"],
                "dc_name":    dc.get("name", ""),
                "operator":   dc.get("operator", ""),
                "county":     dc.get("county", ""),
                "sqft":       dc.get("sqft", np.nan),
                "ring_label": label,
                "geometry":   annulus,
            })

    gdf = gpd.GeoDataFrame(rings, crs=f"EPSG:{EPSG_PROJ}")
    print(f"DC buffer layer: {len(gdf)} rings across {len(dcs)} facilities")
    return gdf


# ── EXPORT TO GEOPACKAGE ──────────────────────────────────────────────────────
# Writes all layers to a single GeoPackage accessible in QGIS.

def export_all(tract_layer, dc_points, dc_buffers):
    print(f"\nExporting to {GPKG_PATH}...")
    layers = {
        "study_tracts": tract_layer,
        "dc_points":    dc_points,
        "dc_buffers":   dc_buffers,
    }

    for county in STUDY_COUNTIES.keys():
        subset = dc_points[dc_points["county"] == county]
        if len(subset) > 0:
            safe = county.lower().replace(" ", "_")
            layers[f"dc_points_{safe}"] = subset

    for name, gdf in layers.items():
        if gdf is None or len(gdf) == 0:
            print(f"  Skipped {name} (empty)")
            continue
        drop = [c for c in gdf.columns if c.startswith("index_")]
        gdf.drop(columns=drop, errors="ignore").to_file(
            GPKG_PATH, layer=name, driver="GPKG"
        )
        print(f"  ✓ {name} ({len(gdf)} features)")

    print(f"\nGeoPackage ready: {GPKG_PATH}")
    print("\nSanity check:")
    print(f"  import geopandas as gpd")
    print(f"  gdf = gpd.read_file('{GPKG_PATH}', layer='study_tracts')")
    print(f"  print(gdf[['med_home_value_2000','med_home_value_2015',")
    print(f"             'med_home_value_2023','county_name']].describe())")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    dcs         = load_study_dcs()
    census_data = fetch_all_data()
    tract_geom  = load_tract_geometries()
    tract_layer = build_tract_layer(tract_geom, census_data)
    dc_buffers  = build_dc_buffers(dcs)
    export_all(tract_layer, dcs, dc_buffers)


if __name__ == "__main__":
    main()