import pandas as pd
from pathlib import Path

ROOT = Path(".")

# Rough bounding box that covers:
# - UW + Laurier
# - Uptown Waterloo
# - Laurel Creek Conservation Area
# - St. Jacobs Market
MIN_LAT = 43.43
MAX_LAT = 43.54
MIN_LON = -80.57
MAX_LON = -80.51


def load_all_stops(root: Path) -> pd.DataFrame:
    """Load GRT and ION stop CSVs that you just downloaded."""
    grt_path = root / "all_grt_stops.csv"
    ion_path = root / "all_ion_stops.csv"

    grt = pd.read_csv(grt_path)
    ion = pd.read_csv(ion_path)

    # Ensure both dataframes share the same columns (union of columns)
    common_cols = sorted(set(grt.columns) | set(ion.columns))

    grt2 = grt.reindex(columns=common_cols)
    ion2 = ion.reindex(columns=common_cols)

    df = pd.concat([grt2, ion2], ignore_index=True)

    # Drop anything without coordinates just in case
    df = df.dropna(subset=["lat", "lon"])
    return df


def filter_to_waterloo_region(df: pd.DataFrame) -> pd.DataFrame:
    """Filter stops to the Waterloo / UW / Laurier / St Jacobs region via bounding box."""
    mask = (
        (df["lat"] >= MIN_LAT)
        & (df["lat"] <= MAX_LAT)
        & (df["lon"] >= MIN_LON)
        & (df["lon"] <= MAX_LON)
    )
    return df[mask].copy()


def main():
    all_stops = load_all_stops(ROOT)
    print(f"Loaded total stops (GRT + ION): {len(all_stops)}")

    waterloo_stops = filter_to_waterloo_region(all_stops)
    print(f"Stops in Waterloo subset area: {len(waterloo_stops)}")

    out_path = ROOT / "waterloo_subset_stops.csv"
    waterloo_stops.to_csv(out_path, index=False)
    print(f"Saved subset to {out_path}")


if __name__ == "__main__":
    main()
