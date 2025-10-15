# `train.nc` Structure

This document summarizes the contents of the competition dataset `train.nc` that should be placed in the same folder as `train_*.py`. This file summarizes the structure of the `.nc` file.

## Overall Shape
- **Dimensions:** `location=83`, `timestamp=2161`, `feature=109`
- **Time span:** hourly records from **2023‑04‑01 00:00 UTC** through **2023‑06‑30 00:00 UTC** (1‑hour resolution)
- **Geography:** all locations correspond to Michigan counties (FIPS prefix `26`)
- **File attributes:** data spans `2023-04-01 00:00:00` to `2023-06-30 00:00:00`

## Coordinates
- `timestamp (2161)`: `datetime64[ns]` increasing hourly sequence covering April–June 2023.
- `location (83)`: string FIPS county identifiers (e.g., `26001`, `26003`, …). Each entry aligns with the same index in `state`.
- `state (83)`: two-character FIPS state codes. Every entry is `"26"` (Michigan).
- `feature (109)`: weather feature identifiers. Maps to standard NWP hazard variables, instrument channel names (`SBTxxx`), or placeholders (`unknown_*`).

## Variables
### `tracked` (`float64`, dims: `location × timestamp`)
- Appears to represent the number of customers/assets under active tracking per county-hour.
- Range: `0` to `922,425`, mean `60026.75`.
- Sparse zero entries (counties/timestamps without tracked assets) but no missing values.

### `out` (`float64`, dims: `location × timestamp`)
- Hourly outage counts (customers impacted) per county.
- Range: `0` to `23,346`, overall mean `45.25`; when outages occur (>0) the conditional mean is `153.25`.
- No missing values. `52,960` county-hours contain non-zero outages.

### `weather` (`float64`, dims: `location × timestamp × feature`)
- 109 weather and hazard predictors co-aligned with the outage grid.
- No missing values; values are already numeric and ready for scaling.
- Example statistics (first five features):
  - `SBT113`: mean `230.96`, std `8.29`, min `0.00`, max `250.18`
  - `SBT114`: mean `277.05`, std `17.73`, min `0.00`, max `305.60`
  - `SBT123`: mean `239.41`, std `9.17`, min `0.00`, max `259.82`
  - `SBT124`: mean `278.65`, std `17.79`, min `0.00`, max `307.90`
  - `aod`: mean `0.00`, std `0.00`, min `0.00`, max `0.00`

#### Feature inventory
The full set of weather feature names is listed below for reference.

| Feature 1 | Feature 2 | Feature 3 |
| --- | --- | --- |
| SBT113 | layth | sulwrf |
| SBT114 | lcc | suswrf |
| SBT123 | lftx | t |
| SBT124 | lftx4 | t2m |
| aod | lsm | tcc |
| bgrun | ltng | tcc_1 |
| blh | max_10si | tcoli |
| cape | mcc | tcolw |
| cape_1 | mdens | tp |
| cfnsf | mslma | u |
| cfrzr | mstav | u10 |
| cicep | orog | unknown |
| cin | pcdb | unknown_1 |
| cnwat | plpl | unknown_2 |
| cpofp | prate | unknown_3 |
| crain | pres | unknown_4 |
| csnow | pres_1 | unknown_5 |
| d2m | pres_2 | unknown_6 |
| fricv | pt | unknown_7 |
| frzr | pwat | unknown_8 |
| fsr | r | unknown_9 |
| gflux | r2 | ustm |
| gh | r_1 | v |
| gh_1 | refc | v10 |
| gh_2 | refd | vbdsf |
| gh_3 | refd_1 | vddsf |
| gh_4 | sde | veg |
| gh_5 | sdlwrf | veril |
| gh_6 | sdswrf | vgtyp |
| gh_7 | sdwe | vis |
| gust | sdwe_1 | vstm |
| hail | sh2 | vucsh |
| hail_1 | siconc | vvcsh |
| hail_2 | slhtf | wz |
| hcc | snowc | wz_1 |
| ishf | sp |  |
| lai | ssrun |  |

## Practical Notes
- All variables share the same `location × timestamp` grid, so joining/pivoting can be done safely on either coordinate pair.
- `tracked` and `out` are float-typed but contain integer-valued counts.
- Weather features span radiation (`sdswrf`, `sulwrf`), precipitation (`prate`, `tp`), temperature (`t`, `t2m`, `SBT***`), wind (`u`, `v`, `u10`, `v10`, `gust`), hydrology (`bgrun`, `ssrun`), and probabilistic hazard indicators (`cfrzr`, `csnow`, `ltng`). Columns prefixed with `unknown` lack metadata in the source file and should be treated as generic continuous predictors.
- Because the dataset covers nearly three months at hourly resolution, memory footprints can be large once expanded; exercise discretion for file read operations.
