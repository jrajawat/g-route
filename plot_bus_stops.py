# plot_bus_stops.py

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

STOPS_CSV = ROOT / "waterloo_subset_stops.csv"
DEMAND_CSV = ROOT / "waterloo_demand_stops.csv"

# Continuous relocation output
CONT_OPT_STOPS_CSV = ROOT / "optimized_stops_continuous.csv"

COVERAGE_RADIUS_M = 600.0


# -------------------------
# Basic helpers
# -------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def nearest_distances(d_lat, d_lon, stops_lat, stops_lon):
    n_dem = len(d_lat)
    dists = np.zeros(n_dem, dtype=float)
    for i in range(n_dem):
        d = haversine_m(d_lat[i], d_lon[i], stops_lat, stops_lon)
        dists[i] = d.min()
    return dists


def load_data():
    stops_df = pd.read_csv(STOPS_CSV)
    demand_df = pd.read_csv(DEMAND_CSV)

    # If there is a weight column, ignore it for plotting purposes
    # We will treat all demand points equally
    return stops_df, demand_df


def meters_to_deg_lat(meters):
    return meters / 111_000.0


def meters_to_deg_lon(meters, lat_deg):
    return meters / (111_000.0 * math.cos(math.radians(lat_deg)))


# ============================================================
# Continuous relocation plots
# ============================================================

def plot_continuous_before_after_and_coverage(stops_df, demand_df):
    """
    Uses optimized_stops_continuous.csv (from the SA script) to plot:
    - Left panel: original stops and coverage
    - Right panel: optimised kept stops and coverage
    Also computes and plots:
    - Average distance before vs after
    - Max distance before vs after

    All demand points are treated equally. No weights are used.
    """
    if not CONT_OPT_STOPS_CSV.exists():
        print("optimized_stops_continuous.csv not found, skipping continuous plots.")
        return

    cont_df = pd.read_csv(CONT_OPT_STOPS_CSV)

    required_cols = {"lat_orig", "lon_orig", "lat", "lon"}
    missing = required_cols - set(cont_df.columns)
    if missing:
        print(f"Continuous file missing columns {missing}, skipping continuous plots.")
        return

    # Original and optimised coordinates per stop
    lat_orig = cont_df["lat_orig"].to_numpy()
    lon_orig = cont_df["lon_orig"].to_numpy()
    lat_opt = cont_df["lat"].to_numpy()
    lon_opt = cont_df["lon"].to_numpy()

    # Kept mask if present (after pruning), otherwise keep all
    if "kept" in cont_df.columns:
        keep_mask = cont_df["kept"].to_numpy().astype(bool)
    else:
        keep_mask = np.ones(len(cont_df), dtype=bool)

    lat_opt_kept = lat_opt[keep_mask]
    lon_opt_kept = lon_opt[keep_mask]

    # Bounds from everything
    all_lat = np.concatenate([lat_orig, lat_opt, demand_df["lat"].to_numpy()])
    all_lon = np.concatenate([lon_orig, lon_opt, demand_df["lon"].to_numpy()])

    pad = 0.002
    x_min = all_lon.min() - pad
    x_max = all_lon.max() + pad
    y_min = all_lat.min() - pad
    y_max = all_lat.max() + pad

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    # Helper to draw coverage circles
    def draw_circles(ax, lats, lons, color):
        for la, lo in zip(lats, lons):
            r_lat = meters_to_deg_lat(COVERAGE_RADIUS_M)
            r_lon = meters_to_deg_lon(COVERAGE_RADIUS_M, la)
            avg_r = (r_lat + r_lon) / 2.0
            ax.add_patch(
                plt.Circle(
                    (lo, la),
                    avg_r,
                    fill=False,
                    edgecolor=color,
                    alpha=0.4,
                    linewidth=0.8,
                )
            )

    # Demand points all same size since we are ignoring weights
    demand_lat = demand_df["lat"].to_numpy()
    demand_lon = demand_df["lon"].to_numpy()

    # Panel 1: original
    ax0 = axes[0]
    ax0.scatter(
        demand_lon, demand_lat,
        s=30,
        c="blue",
        alpha=0.7,
        label="Demand points",
    )
    ax0.scatter(
        lon_orig, lat_orig,
        s=20,
        c="gray",
        alpha=0.9,
        label="Original stops",
    )
    draw_circles(ax0, lat_orig, lon_orig, "green")
    ax0.set_title("Original stops and coverage")
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.set_xlim(x_min, x_max)
    ax0.set_ylim(y_min, y_max)
    ax0.set_aspect("equal")
    ax0.legend()

    # Panel 2: optimised kept
    ax1 = axes[1]
    ax1.scatter(
        demand_lon, demand_lat,
        s=30,
        c="blue",
        alpha=0.7,
        label="Demand points",
    )
    ax1.scatter(
        lon_opt_kept, lat_opt_kept,
        s=30,
        c="red",
        edgecolors="k",
        label="Optimised kept stops",
    )
    draw_circles(ax1, lat_opt_kept, lon_opt_kept, "green")
    ax1.set_title("Optimised stops and coverage (continuous relocation)")
    ax1.set_xlabel("Longitude")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect("equal")
    ax1.legend()

    fig.tight_layout()
    out_path = PLOTS / "4_continuous_before_after_coverage.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Saved:", out_path)

    # --------------------------------------------------
    # Distances before vs after
    # --------------------------------------------------
    d_lat = demand_lat
    d_lon = demand_lon

    # Before uses original stops
    before = nearest_distances(d_lat, d_lon, lat_orig, lon_orig)

    # After uses optimised kept stops
    after = nearest_distances(d_lat, d_lon, lat_opt_kept, lon_opt_kept)

    avg_before = float(before.mean())
    avg_after = float(after.mean())
    max_before = float(before.max())
    max_after = float(after.max())

    print("\nContinuous relocation distance metrics (unweighted):")
    print(f"  Average distance before: {avg_before:.1f} m")
    print(f"  Average distance after:  {avg_after:.1f} m")
    print(f"  Max distance before:     {max_before:.1f} m")
    print(f"  Max distance after:      {max_after:.1f} m")

    # Bar plot for average and max distances
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    labels = ["Average distance (m)", "Max distance (m)"]
    x = np.arange(len(labels))
    width = 0.35

    before_vals = [avg_before, max_before]
    after_vals = [avg_after, max_after]

    ax2.bar(x - width / 2, before_vals, width, label="Before")
    ax2.bar(x + width / 2, after_vals, width, label="After")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Distance (m)")
    ax2.set_title("Continuous relocation average and max distance")

    for i in range(len(labels)):
        ax2.text(
            x[i] - width / 2,
            before_vals[i],
            f"{before_vals[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax2.text(
            x[i] + width / 2,
            after_vals[i],
            f"{after_vals[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(PLOTS / "5_continuous_avg_and_max.png", dpi=300)
    plt.close(fig2)
    print("Saved:", PLOTS / "5_continuous_avg_and_max.png")

    # Histogram comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(before.max(), after.max()), 35)
    ax3.hist(before, bins=bins, alpha=0.5, label="Before")
    ax3.hist(after, bins=bins, alpha=0.5, label="After")
    ax3.set_xlabel("Walking distance (m)")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of walking distances before and after")
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(PLOTS / "6_continuous_distance_hist.png", dpi=300)
    plt.close(fig3)
    print("Saved:", PLOTS / "6_continuous_distance_hist.png")

    # CDF comparison
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    def plot_cdf(values, label):
        sorted_vals = np.sort(values)
        y = np.linspace(0.0, 1.0, len(sorted_vals))
        ax4.plot(sorted_vals, y, label=label, linewidth=2)

    plot_cdf(before, "Before")
    plot_cdf(after, "After")

    ax4.set_xlabel("Walking distance (m)")
    ax4.set_ylabel("Cumulative fraction")
    ax4.set_title("CDF of walking distances before and after")
    ax4.grid(alpha=0.25)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(PLOTS / "7_continuous_distance_cdf.png", dpi=300)
    plt.close(fig4)
    print("Saved:", PLOTS / "7_continuous_distance_cdf.png")


# ============================================================
# Main
# ============================================================

def main():
    stops_df, demand_df = load_data()
    print(f"Loaded {len(stops_df)} stops and {len(demand_df)} demand points.")

    # Only continuous relocation plots
    plot_continuous_before_after_and_coverage(stops_df, demand_df)

    print("All plots saved to:", PLOTS)


if __name__ == "__main__":
    main()
