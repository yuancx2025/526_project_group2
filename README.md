# Airline Price Analysis and Market Structure

This is the repo for CS 526 group2's project. This repository contains three Jupyter notebooks that implement our analysis of airline ticket prices, market structure classification, route‑level mapping, and predictive modeling. It also includes a smaller, route‑level dataset used for fixed‑effects regressions.

- Notebooks
  - [Flight_Prices_Analysis.ipynb](Flight_Prices_Analysis.ipynb): Large ticket‑level EDA on the Kaggle “flightprices” dataset, aggregations and market HHI.
  - [ds_kaggle.ipynb](ds_kaggle.ipynb): Data quality checks, HHI classification on a 20k sample, route maps, and Ridge regression models.
  - [Data_science.ipynb](Data_science.ipynb): Route‑year panel analysis using a curated route dataset (HHI vs fares/inequality) plus route visualizations.

- Data
  - Kaggle dataset: dilwong/flightprices (downloaded in code).
  - [itineraries_sample.csv](itineraries_sample.csv): Sample of 20k tickets.
  - [KG_cost_small.csv](KG_cost_small.csv): Route‑level panel for macro analysis.

## Environment

The notebooks are designed for Google Colab, but run locally with the same Python packages.

Key packages installed within notebooks:
- kaggle, pandas, pyarrow (for data ingestion and Parquet)
- matplotlib, seaborn, plotly (viz)
- scikit‑learn (modeling)
- geopandas, geodatasets, shapely (mapping)
- linearmodels, statsmodels (panel regressions)

## 1) Ticket‑level EDA and route HHI (Flight_Prices_Analysis.ipynb)

1. Download and persist Kaggle data
   - Use Kaggle API to download “dilwong/flightprices”.
   - Convert CSV to 83 Parquet parts and store on Google Drive to survive session restarts.

2. Aggregations and plots (computed by reading all Parquet parts)
   - Average fares by airline
     - Group by segmentsAirlineName; compute sum(baseFare), sum(totalFare), and counts; derive averages by dividing sums by counts.
     - Optionally filter to single‑carrier trips (exclude strings containing “||”).
   - Average fares vs days until flight
     - Compute daysUntilFlight = flightDate − searchDate.
     - Group by daysUntilFlight and plot mean baseFare and mean totalFare vs days.
   - Average fares vs seats remaining
     - Group by seatsRemaining, drop the 0 bin, plot mean baseFare and mean totalFare vs seats.
   - Average fares by departure/arrival airport
     - Group by startingAirport and destinationAirport, plot stacked bars: baseFare and (totalFare − baseFare).
   - Average fares vs total travel distance
     - Group by totalTravelDistance, drop the 0 bin, scatter plot mean totalFare vs distance.

3. Route market structure (HHI)
   - Build per‑route airline counts (filter to single‑carrier itineraries by excluding “||” in segmentsAirlineName).
   - Compute route totals, per‑airline market share, and $HHI=\sum s_i^2$.
   - Classify market type per route:
     - HHI > 0.6 → Monopoly
     - 0.2 < HHI ≤ 0.6 → Oligopoly
     - 0 < HHI ≤ 0.2 → Competitive
   - Export/display the per‑route table with HHI and Market Type.

See notebook: [Flight_Prices_Analysis.ipynb](Flight_Prices_Analysis.ipynb)

## 2) Sample data QA, HHI, mapping, and modeling (ds_kaggle.ipynb)

1. Load sample
   - Mount Drive, load [itineraries_sample.csv](itineraries_sample.csv) into a DataFrame.

2. Market structure via HHI on sample
   - Group by startingAirport, destinationAirport, segmentsAirlineCode to get counts and within‑route shares.
   - Compute $HHI=\sum s_i^2$ per route.
   - Classify per route:
     - HHI > 0.25 → Monopoly
     - 0.15 < HHI ≤ 0.25 → Oligopoly
     - Else → Competitive
   - Merge back to the sample.

3. Data quality checks (pre_checks pipeline)
   - Duplicates: legId uniqueness.
   - Fares, booleans, seats: positivity/order checks, boolean coercion, seat summary.
   - Time integrity: parse dates, enforce searchDate ≤ flightDate, booking window stats.
   - Segment consistency: segment leg counts across columns, endpoint coherence (segment vs route endpoints).
   - Duration and distance alignment: totals vs sum of segment values (±10% tolerance).
   - Price coherence: refundable uplift, nonstop premium, cabin medians.
   - Missingness: per‑column missing rates.
   - Coverage & HHI: distribution of MarketType.
   - Outliers: IQR‑based on totalFare.

4. Route map (Plotly Scattergeo)
   - Build route lat/lon by merging IATA codes to coordinates from OpenFlights airports dataset.
   - Sample routes to avoid clutter and draw geodesic lines colored by MarketType.

5. Ridge regression models (log totalFare)
   - Features and engineering:
     - booking_window = days between flightDate and searchDate (clipped to [0,365]).
     - num_legs from “||” count in segments time fields.
     - route_distance from totalTravelDistance else sum(segmentsDistance); winsorize [1%, 99%].
     - Clean booleans to 0/1; seatsRemaining numeric.
   - Baseline model:
     - Numeric: booking_window, route_distance, num_legs, seatsRemaining.
     - Categorical: startingAirport, destinationAirport, segmentsAirlineCode, isNonStop, isRefundable, isBasicEconomy.
     - Pipeline: ColumnTransformer(impute+scale numeric; impute+OneHot categorical), RidgeCV(alphas logspaced, 5‑fold), time split by flightDate (80/20).
     - Report alpha, Train/Test R², median absolute error (in $), MAPE.
     - Inspect coefficients and plot top positive/negative.
   - Simplified carrier model:
     - Parse carrier lists from segmentsAirlineCode, derive: first/last carrier, n_unique_carriers, n_carrier_changes, has_ulcc (NK/F9/G4), has_lcc (WN/B6/AS), and presence flags for top M carriers from train only.
     - Restrict OneHot on low‑cardinality cats (min_frequency threshold), keep presence flags numeric.
     - Same pipeline/training/evaluation as baseline.

See notebook: [ds_kaggle.ipynb](ds_kaggle.ipynb)

## 3) Route‑year panel and visualizations (Data_science.ipynb)

1. Load route panel and enrich with coordinates
   - Read [KG_cost_small.csv](KG_cost_small.csv) from GitHub.
   - Load OpenFlights airports (airports.dat), select IATA, lat, lon, country, city.
   - Merge origin/destination coordinates into the panel; keep U.S. to form data_us.

2. 2005 route snapshot and market type
   - Build routes_2005 by grouping data_us (year==2005) on origin, dest; carry coords, mean HHI_route, and endpoint populations.
   - Classify market_type from HHI_route using bins:
     - competitive: HHI ≤ 0.2
     - oligopoly: 0.2 < HHI ≤ 0.6
     - monopoly: HHI > 0.6

3. Visualizations
   - Countplot: number of routes by market_type (ordered competitive → oligopoly → monopoly).
   - Geo map (GeoPandas):
     - Sample ~15% of routes for readability.
     - Build LineString geometries (origin→dest), plot over US basemap; color by market_type.
   - City size interaction:
     - Define big city indicators using 80th percentile of pop_o/pop_d; route_city_type = origin_bigcity + dest_bigcity in {0,1,2}.
     - Faceted distribution of market_type counts by route_city_type with per‑bar labels.
   - Percentage faceted chart:
     - Compute within‑facet percentages of market_type and plot as bars with percentage labels.

4. Panel fixed‑effects regressions
   - Set index: route, year.
   - Estimate three PanelOLS models with route fixed effects (EntityEffects) and year fixed effects (TimeEffects), cluster standard errors by entity:
     - meanfare ~ HHI_route + FE
     - minfare  ~ HHI_route + FE
     - gini     ~ HHI_route + FE
   - Print summaries to assess relationship between concentration (HHI_route) and prices/inequality.

See notebook: [Data_science.ipynb](Data_science.ipynb)

## Reproducing the workflow

- If using Colab:
  - Open the notebooks with the “Open in Colab” badges at the top of each file.
  - In [Flight_Prices_Analysis.ipynb](Flight_Prices_Analysis.ipynb), run the Kaggle download and Parquet conversion once. Subsequent cells read all Parquet parts.
  - In [ds_kaggle.ipynb](ds_kaggle.ipynb), load [itineraries_sample.csv](itineraries_sample.csv), run pre_checks, mapping, and modeling cells in order.
  - In [Data_science.ipynb](Data_science.ipynb), run from top to build merges, visuals, and the panel regressions.

- If running locally:
  - Install the dependencies listed above.
  - Replace Google Drive/Colab paths with local paths as needed and skip Drive mounts.

## Notes

- HHI thresholds differ by notebook:
  - [Flight_Prices_Analysis.ipynb](Flight_Prices_Analysis.ipynb): Monopoly > 0.6; Oligopoly (0.2, 0.6]; Competitive (0, 0.2].
  - [ds_kaggle.ipynb](ds_kaggle.ipynb): Monopoly > 0.25; Oligopoly (0.15, 0.25]; Competitive ≤ 0.15.
- OpenFlights airports reference is pulled live in both [Data_science.ipynb](Data_science.ipynb) and [ds_kaggle.ipynb](ds_kaggle.ipynb) to geocode IATA.

## License

See [LICENSE](LICENSE).
