import requests
import pandas as pd
from pathlib import Path

# ArcGIS Open Data endpoints
GRT_URL = (
    "https://utility.arcgis.com/usrsvcs/servers/"
    "52c4134809a94f85b31a2e9553de1358/rest/services/OpenData/OpenData/MapServer/3/query"
)

ION_URL = (
    "https://utility.arcgis.com/usrsvcs/servers/"
    "f063d1fb147847f796ce8c024e117419/rest/services/OpenData/OpenData/MapServer/5/query"
)

# Save into root folder
ROOT = Path(".")

def fetch_arcgis_features(url: str):
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": 4326,
        "f": "json",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    print(f"{url} -> {len(features)} features")
    return features


def grt_features_to_df(features):
    rows = []
    for feat in features:
        attrs = feat.get("attributes", {})
        geom = feat.get("geometry", {}) or {}

        lon = attrs.get("Longitude", geom.get("x"))
        lat = attrs.get("Latitude", geom.get("y"))

        rows.append(
            {
                "source": "GRT",
                "stop_id": attrs.get("StopID"),
                "name": f"{attrs.get('Street', '')} at {attrs.get('CrossStreet', '')}".strip(),
                "street": attrs.get("Street"),
                "cross_street": attrs.get("CrossStreet"),
                "municipality": attrs.get("Municipality"),
                "status": attrs.get("Status"),
                "lon": lon,
                "lat": lat,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lat", "lon"])
    return df


def ion_features_to_df(features):
    rows = []
    for feat in features:
        attrs = feat.get("attributes", {})
        geom = feat.get("geometry", {}) or {}

        lon = geom.get("x")
        lat = geom.get("y")

        rows.append(
            {
                "source": "ION",
                "stop_id": attrs.get("StopName"),
                "name": attrs.get("StopName"),
                "municipality": attrs.get("Municipality"),
                "direction": attrs.get("StopDirection"),
                "status": attrs.get("StopStatus"),
                "stage1": attrs.get("Stage1"),
                "stage2": attrs.get("Stage2"),
                "phase": attrs.get("Phase"),
                "lon": lon,
                "lat": lat,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lat", "lon"])
    return df


def main():
    # Download stops
    grt_features = fetch_arcgis_features(GRT_URL)
    ion_features = fetch_arcgis_features(ION_URL)

    grt_df = grt_features_to_df(grt_features)
    ion_df = ion_features_to_df(ion_features)

    # Save to root folder
    grt_path = ROOT / "all_grt_stops.csv"
    ion_path = ROOT / "all_ion_stops.csv"

    grt_df.to_csv(grt_path, index=False)
    ion_df.to_csv(ion_path, index=False)

    print(f"Saved {len(grt_df)} GRT stops -> {grt_path}")
    print(f"Saved {len(ion_df)} ION stops -> {ion_path}")


if __name__ == "__main__":
    main()
