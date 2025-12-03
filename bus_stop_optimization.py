#!/usr/bin/env python
"""
Continuous relocation optimisation (equal treatment, no weights).

Behaviour:
- Uses the same initial set of stops.
- Allows each stop to move within a limited radius using simulated annealing.
- Objective is to minimise the unweighted sum of distances from:
    - all demand points, and
    - all stops themselves (treated as points that also care about access)
  to the nearest stop.
- Then greedily prunes stops that are redundant, but only if this does NOT
  worsen:
    - the average distance, or
    - the maximum distance
  relative to the simulated annealing solution, when evaluated over all
  points (demand + stops).
"""

import math
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# Input files
STOPS_CSV = ROOT / "waterloo_subset_stops.csv"
DEMAND_CSV = ROOT / "waterloo_demand_stops.csv"

# Output files
OPT_STOPS_CSV = ROOT / "optimized_stops_continuous.csv"
OPT_STOPS_GEOJSON = ROOT / "optimized_stops_continuous.geojson"

# General parameters
COVERAGE_RADIUS_M = 600.0   # for reporting only
R_MAX_M = 150.0             # max movement allowed from original stop (meters)
STEP_SIZE_M = 40.0          # typical movement step size in a proposal (meters)

# Simulated annealing parameters
SA_T0 = 1.0
SA_T_MIN = 1e-3
SA_ALPHA = 0.95
SA_ITERS_PER_T = 250        # iterations per temperature


# ============================================================
# Geometry helpers
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Great circle distance between (lat1, lon1) and (lat2, lon2) in meters.
    lat* and lon* can be numpy arrays, broadcasting is supported.
    """
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2) - np.radians(lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def latlon_step(lat_deg, step_m):
    """
    Convert a random polar step of radius step_m (0..step_m)
    into dlat and dlon in degrees at a given latitude.
    """
    angle = np.random.uniform(0.0, 2.0 * math.pi)
    r = np.random.uniform(0.0, step_m)

    d_north = r * math.sin(angle)
    d_east = r * math.cos(angle)

    dlat = d_north / 111000.0
    lon_scale = 111000.0 * math.cos(math.radians(lat_deg))
    if abs(lon_scale) < 1e-6:
        dlon = 0.0
    else:
        dlon = d_east / lon_scale

    return dlat, dlon


# ============================================================
# Data loading and metrics
# ============================================================

def load_data():
    if not STOPS_CSV.exists():
        raise FileNotFoundError(f"{STOPS_CSV} not found")

    if not DEMAND_CSV.exists():
        raise FileNotFoundError(f"{DEMAND_CSV} not found")

    stops_df = pd.read_csv(STOPS_CSV)
    required_cols_stops = {"stop_id", "name", "lat", "lon"}
    missing = required_cols_stops - set(stops_df.columns)
    if missing:
        raise ValueError(f"{STOPS_CSV} missing columns: {missing}")

    demand_df = pd.read_csv(DEMAND_CSV)
    required_cols_dem = {"name", "lat", "lon"}
    missing = required_cols_dem - set(demand_df.columns)
    if missing:
        raise ValueError(f"{DEMAND_CSV} missing columns: {missing}")

    # Ignore any weight columns that might exist.
    if "weight" in demand_df.columns:
        demand_df = demand_df.drop(columns=["weight"])

    stops_df = stops_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    demand_df = demand_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    return stops_df, demand_df


def compute_total_distance(
    demand_lat, demand_lon,
    stop_lat, stop_lon,
):
    """
    Compute:
      - sum of nearest distances from each demand point to the nearest stop
      - vector of nearest distances per demand point

    All points are treated equally (unweighted).
    """
    n_dem = len(demand_lat)
    n_stop = len(stop_lat)
    dist_mat = np.zeros((n_dem, n_stop), dtype=float)

    for i in range(n_dem):
        dist_mat[i, :] = haversine_m(
            demand_lat[i], demand_lon[i],
            stop_lat, stop_lon,
        )

    nearest = dist_mat.min(axis=1)
    total = float(nearest.sum())
    return total, nearest


def summarise_metrics(nearest_dist):
    """
    Return a dict of summary metrics given nearest distances.
    All points are treated equally.
    """
    avg = float(nearest_dist.mean())
    max_d = float(nearest_dist.max())
    coverage_fraction = float((nearest_dist <= COVERAGE_RADIUS_M).mean())

    return {
        "avg_m": avg,
        "max_m": max_d,
        "coverage_fraction": coverage_fraction,
    }


# ============================================================
# Simulated annealing
# ============================================================

def propose_new_position(j, stop_lat, stop_lon, orig_lat, orig_lon):
    """
    Propose a new position for stop j, constrained to be within
    R_MAX_M of the original coordinate.
    """
    lat_old = stop_lat[j]
    lon_old = stop_lon[j]
    lat0 = orig_lat[j]
    lon0 = orig_lon[j]

    for _ in range(10):
        dlat, dlon = latlon_step(lat_old, STEP_SIZE_M)
        lat_new = lat_old + dlat
        lon_new = lon_old + dlon

        # Constrain to stay near the original position
        dist_from_orig = float(haversine_m(lat_new, lon_new, lat0, lon0))
        if dist_from_orig <= R_MAX_M:
            return lat_new, lon_new

    # If repeated attempts fail, keep old position
    return lat_old, lon_old


def simulated_annealing(
    all_lat, all_lon,          # points whose distance we care about
    stop_lat_init, stop_lon_init,
    orig_lat, orig_lon,
):
    """
    Run simulated annealing to optimise stop coordinates.

    Objective: minimise unweighted total distance from all points (all_lat, all_lon)
    to the nearest stop.
    """
    stop_lat = stop_lat_init.copy()
    stop_lon = stop_lon_init.copy()

    best_total, nearest = compute_total_distance(
        all_lat, all_lon,
        stop_lat, stop_lon,
    )
    best_lat = stop_lat.copy()
    best_lon = stop_lon.copy()

    current_total = best_total
    print(f"Initial total distance: {current_total:.1f} m")

    T = SA_T0
    iters = 0
    t_start = time.time()

    while T > SA_T_MIN:
        for _ in range(SA_ITERS_PER_T):
            iters += 1

            # Pick a random stop to move
            j = np.random.randint(0, len(stop_lat))

            lat_new, lon_new = propose_new_position(
                j, stop_lat, stop_lon, orig_lat, orig_lon,
            )

            # Save old coordinates
            old_lat_j = stop_lat[j]
            old_lon_j = stop_lon[j]

            # Apply candidate move
            stop_lat[j] = lat_new
            stop_lon[j] = lon_new

            # Compute new objective
            new_total, _ = compute_total_distance(
                all_lat, all_lon,
                stop_lat, stop_lon,
            )

            delta = new_total - current_total

            # Acceptance rule
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / T)
                accept = np.random.rand() < prob

            if accept:
                current_total = new_total
                if new_total < best_total:
                    best_total = new_total
                    best_lat = stop_lat.copy()
                    best_lon = stop_lon.copy()
            else:
                # revert
                stop_lat[j] = old_lat_j
                stop_lon[j] = old_lon_j

        T *= SA_ALPHA

    elapsed = time.time() - t_start
    print(f"SA finished in {iters} iterations, {elapsed:.2f} s")
    print(f"Best total distance: {best_total:.1f} m")

    info = {
        "iterations": iters,
        "time_sec": elapsed,
        "best_total": best_total,
    }
    return best_lat, best_lon, info


# ============================================================
# Pruning redundant stops relative to SA solution
# ============================================================

def prune_stops(
    all_lat, all_lon,          # points whose distance we care about
    lat_opt, lon_opt,
    ref_metrics,
    eps=1e-6,
):
    """
    Greedily remove stops if doing so does NOT make things worse
    than the reference SA solution on average distance and max distance.

    Criteria for accepting a removal:
      - avg_m <= ref avg_m + eps
      - max_m <= ref max_m + eps

    lat_opt and lon_opt are the SA optimised coordinates for all stops.
    We keep them full length and maintain a boolean mask of which stops
    are actually active.
    """
    n_stop = len(lat_opt)
    active = np.ones(n_stop, dtype=bool)

    ref_avg = ref_metrics["avg_m"]
    ref_max = ref_metrics["max_m"]

    def metrics_for_mask(mask):
        lat_a = lat_opt[mask]
        lon_a = lon_opt[mask]
        _, nearest = compute_total_distance(
            all_lat, all_lon,
            lat_a, lon_a,
        )
        return summarise_metrics(nearest)

    # Metrics for full optimised network (before pruning)
    opt_metrics_full = metrics_for_mask(active)
    print("\nBefore pruning (all points as demand):")
    print(f"  Avg distance: {opt_metrics_full['avg_m']:.1f} m")
    print(f"  Max dist:     {opt_metrics_full['max_m']:.1f} m")
    print(f"  Coverage <= {COVERAGE_RADIUS_M:.0f} m: {opt_metrics_full['coverage_fraction']*100:.1f}%")

    changed = True
    iteration = 0

    while changed:
        changed = False
        iteration += 1
        print(f"\nPruning pass {iteration}...")

        for j in range(n_stop):
            if not active[j]:
                continue

            trial_mask = active.copy()
            trial_mask[j] = False

            met = metrics_for_mask(trial_mask)

            if (
                met["avg_m"] <= ref_avg + eps
                and met["max_m"] <= ref_max + eps
            ):
                # Accept removal
                active = trial_mask
                print(
                    f"  Removed stop index {j}: "
                    f"avg={met['avg_m']:.1f} m, "
                    f"max={met['max_m']:.1f} m"
                )
                changed = True

        if not changed:
            print("  No further removable stops found.")

    final_metrics = metrics_for_mask(active)
    print("\nAfter pruning (relative to SA solution):")
    print(f"  Kept {active.sum()} stops out of {n_stop}")
    print(
        f"  Avg distance: {final_metrics['avg_m']:.1f} m "
        f"(SA {ref_avg:.1f} m)"
    )
    print(
        f"  Max dist:     {final_metrics['max_m']:.1f} m "
        f"(SA {ref_max:.1f} m)"
    )
    print(
        f"  Coverage <= {COVERAGE_RADIUS_M:.0f} m: "
        f"{final_metrics['coverage_fraction']*100:.1f}%"
    )

    return active, final_metrics


# ============================================================
# Outputs
# ============================================================

def save_optimised_stops(stops_df, lat_opt, lon_opt, keep_mask):
    """
    Save a CSV with original and optimised positions and a keep flag.
    Also save a GeoJSON with only kept stops.
    """
    out_df = stops_df.copy()
    out_df["lat_orig"] = out_df["lat"]
    out_df["lon_orig"] = out_df["lon"]
    out_df["lat"] = lat_opt
    out_df["lon"] = lon_opt

    move_dist = haversine_m(
        out_df["lat_orig"].to_numpy(),
        out_df["lon_orig"].to_numpy(),
        lat_opt,
        lon_opt,
    )
    out_df["move_distance_m"] = move_dist

    # Mark which stops are actually kept in service after pruning
    out_df["kept"] = keep_mask.astype(int)

    out_df.to_csv(OPT_STOPS_CSV, index=False)
    print(f"Saved optimised stops to {OPT_STOPS_CSV}")

    # Simple GeoJSON for mapping, only export kept stops
    features = []
    for _, row in out_df[out_df["kept"] == 1].iterrows():
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]],
            },
            "properties": {
                "stop_id": row["stop_id"],
                "name": row["name"],
                "move_m": float(row["move_distance_m"]),
            },
        }
        features.append(feat)

    fc = {"type": "FeatureCollection", "features": features}
    with open(OPT_STOPS_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    print(f"Saved GeoJSON to {OPT_STOPS_GEOJSON}")


# ============================================================
# Main
# ============================================================

def main():
    stops_df, demand_df = load_data()
    print(f"Loaded {len(stops_df)} stops and {len(demand_df)} demand points.")

    # Facility locations (things we can move and prune)
    stop_lat_orig = stops_df["lat"].to_numpy()
    stop_lon_orig = stops_df["lon"].to_numpy()

    # Demand points from the separate demand file
    demand_lat = demand_df["lat"].to_numpy()
    demand_lon = demand_df["lon"].to_numpy()

    # Points whose distance we care about:
    # all demand points + all stops treated as points too.
    all_lat = np.concatenate([demand_lat, stop_lat_orig])
    all_lon = np.concatenate([demand_lon, stop_lon_orig])

    # Baseline (original stop positions), evaluated on all points
    base_total, base_nearest = compute_total_distance(
        all_lat, all_lon,
        stop_lat_orig, stop_lon_orig,
    )
    base_metrics = summarise_metrics(base_nearest)

    print("\nBaseline metrics (all demand + all stops as demand):")
    print(f"  Avg distance: {base_metrics['avg_m']:.1f} m")
    print(f"  Max dist:     {base_metrics['max_m']:.1f} m")
    print(
        f"  Coverage <= {COVERAGE_RADIUS_M:.0f} m: "
        f"{base_metrics['coverage_fraction']*100:.1f}%"
    )

    # Run SA to relocate stops (all stops active)
    lat_opt, lon_opt, info = simulated_annealing(
        all_lat, all_lon,
        stop_lat_orig, stop_lon_orig,
        stop_lat_orig, stop_lon_orig,
    )

    opt_total, opt_nearest = compute_total_distance(
        all_lat, all_lon,
        lat_opt, lon_opt,
    )
    opt_metrics = summarise_metrics(opt_nearest)

    print("\nOptimised metrics (before pruning, all points):")
    print(f"  Avg distance: {opt_metrics['avg_m']:.1f} m")
    print(f"  Max dist:     {opt_metrics['max_m']:.1f} m")
    print(
        f"  Coverage <= {COVERAGE_RADIUS_M:.0f} m: "
        f"{opt_metrics['coverage_fraction']*100:.1f}%"
    )

    improvement = base_metrics["avg_m"] - opt_metrics["avg_m"]
    print(
        f"\nChange in average distance (before pruning): "
        f"{improvement:.1f} m"
    )

    # Prune redundant stops while keeping average and max distance
    # at least as good as the SA solution on all points
    keep_mask, pruned_metrics = prune_stops(
        all_lat, all_lon,
        lat_opt, lon_opt,
        opt_metrics,
    )

    # Save results
    save_optimised_stops(stops_df, lat_opt, lon_opt, keep_mask)


if __name__ == "__main__":
    main()
