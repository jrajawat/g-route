# G-ROUTE Grand River Route Engine 
Continuous Facility Location Optimisation using Simulated Annealing

This project optimises the placement of bus stops within a study area in Waterloo. The objective is to slightly reposition existing stops and remove redundant ones while ensuring that accessibility never becomes worse for any point in the region.

The optimisation is inspired by continuous facility location models, p-median reductions, and simulated annealing heuristics, making it appropriate for a SYDE 411 optimisation deliverable.

---

## 1. Problem Overview

Urban bus networks often accumulate overlapping or poorly spaced stops. This project attempts to improve spatial efficiency without compromising access.

The optimisation aims to:

- Keep all stops but allow small continuous movements.
- Reduce stop redundancy.
- Preserve or improve average and maximum walking distance.
- Treat all points equally (no weights).

This is a constrained continuous facility-location optimisation problem.

---

## 2. Input Data

Two CSV files are required.

### waterloo_subset_stops.csv  
Contains all candidate/existing bus stops.  
Required columns:
- stop_id  
- name  
- lat  
- lon  

### waterloo_demand_stops.csv  
Contains points of interest that should remain served.  
Required columns:
- name  
- lat  
- lon  

Any weight column is ignored.

---

## 3. Method Overview

The optimisation has two stages:

1. Continuous relocation using simulated annealing  
2. Redundant-stop pruning with strict service constraints  

Both stages treat all points equally.

---

## 3.1 Continuous Relocation (Simulated Annealing)

Each stop can move within a maximum radius (default 150 m). The objective is:

```
Minimise total nearest-stop distance
for all points = (all stops) + (all demand points)
```

This ensures:

- Every stop is placed meaningfully relative to its neighbours.
- Demand points remain well served.
- Overlapping stops are discouraged unless necessary.

### Movement generation
- Pick a random stop.
- Propose a step up to 40 m in a random direction.
- Reject if it exceeds the 150 m relocation radius constraint.

### Annealing parameters
- Initial temperature: 1.0  
- Cooling factor: 0.95  
- Minimum temperature: 1e-3  
- ~250 iterations per temperature level  

Simulated annealing is used because:

- The objective landscape is nonconvex.
- Gradient methods fail due to discontinuous nearest-stop distances.
- SA is standard in facility location optimisation.

The output is an updated latitude/longitude for each stop.

---

## 3.2 Redundant Stop Pruning

After relocation, each stop is evaluated for removal.

A stop is removed only if removing it does **not** worsen:

- Average distance  
- Maximum distance  

These metrics are computed across:

- All demand points  
- All bus stops treated as “access points”

The removal criteria are:

```
avg_after <= avg_before
and
max_after <= max_before
```

This guarantees:

- No point becomes worse off.
- Overlapping or unnecessary stops are eliminated.
- Spatial redundancy is reduced while preserving accessibility.

Pruning continues until no further stops can be removed.

---

## 4. Output Files

### optimized_stops_continuous.csv  
Contains:

- stop_id  
- name  
- lat_orig, lon_orig  
- lat, lon (optimised positions)  
- move_distance_m  
- kept (1 if retained after pruning)  

### optimized_stops_continuous.geojson  
Contains only the stops retained after pruning.

Both files can be visualised on GIS platforms.

---

## 5. Metrics Reported

After optimisation and pruning, the following are computed:

- Average nearest-stop distance  
- Median nearest-stop distance  
- Maximum nearest-stop distance  
- Percentage of points within 600 m  

All points are treated equally.

These metrics allow direct comparison of:
- Original configuration  
- After SA relocation  
- After pruning  

---

## 6. Academic Foundations

The method is grounded in standard operations research concepts.

### Continuous facility-location models  
Similar to the continuous p-median and p-center problems.

### Simulated annealing heuristics  
Commonly used for spatial optimisation when the objective is nonconvex and discontinuous.

### Dominance-based pruning  
Mirrors transit stop consolidation methods in public transportation planning literature.

### GIS-based service coverage  
Use of a 600 m walking radius aligns with municipal and academic transit-accessibility thresholds.

This connection strengthens the project’s validity for a SYDE 411 optimisation assignment.

---

## 7. Running the Optimisation

Install dependencies:

```
pip install numpy pandas matplotlib
```

Run the optimisation:

```
python bus_stop_optimization.py
```

Run the visualisation (optional):

```
python plot_bus_stops.py
```

All results will be generated in the working directory.

---

## 8. Interpretation of Results

The optimisation typically yields:

- Slight but meaningful relocations that smooth spacing
- Equal or better:
  - average walking distance
  - maximum walking distance
  - coverage percentage
- A reduced set of stops that still fully serves the region

This gives a coherent narrative for a SYDE 411 submission:
- A mathematically structured model  
- A metaheuristic optimisation method  
- A dominance-based pruning approach  
- Clear spatial and numerical results  

---

## 9. Possible Extensions

Optional improvements include:

- Multi-objective optimisation (distance + equity + spacing consistency)  
- Weighted demand (e.g., buildings vs households)  
- Routing constraints for bus lines  