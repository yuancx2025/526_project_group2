# Airline Price Analysis and Market Structure

This is the repo for CS 526 group2's project. This repository contains multiple Jupyter notebooks that implement our analysis of airline ticket prices, market structure classification, route‑level mapping, and predictive modeling using various machine learning techniques. It also includes route‑level datasets used for market structure analysis and fixed‑effects regressions.

## Repository Structure

```
526_project_group2/
├── competetive_structual_analysis/
│   └── Data_science.ipynb          # Market structure & panel regression analysis
├── price_prediction/
│   ├── Flight_Prices_Analysis.ipynb # EDA and market HHI computation
│   ├── ridge.ipynb                  # Ridge regression price prediction
│   ├── lasso.ipynb                  # Lasso regression price prediction
│   └── xgboost.ipynb                # XGBoost price prediction
└── data/
    ├── sample_20k.csv               # Sample of 20k flight tickets
    └── KG_cost_small.csv            # Route‑level panel dataset
```

## Notebooks Overview

### Market Structure Analysis
- **[competetive_structual_analysis/Data_science.ipynb](competetive_structual_analysis/Data_science.ipynb)**: Route‑year panel analysis using the route dataset, examining relationships between market concentration (HHI) and fare levels/inequality, with U.S. route visualizations.
- **[competetive_structual_analysis/PCA.ipynb](competetive_structual_analysis/PCA.ipynb)**:
This notebook performs a Principal Component Analysis (PCA) on a 5,000-example sample of flight itineraries. It loads the dataset, preprocesses the relevant features, applies PCA, and visualizes the first two principal components. The resulting plot shows that the number of flight legs is the main source of variation in the data.

### Price Prediction & EDA
- **[price_prediction/Flight_Prices_Analysis.ipynb](price_prediction/Flight_Prices_Analysis.ipynb)**: Large ticket‑level exploratory data analysis on the Kaggle "flightprices" dataset, including fare aggregations by various dimensions and HHI market structure classification.
- **[price_prediction/ridge.ipynb](price_prediction/ridge.ipynb)**: Ridge regression model for flight price prediction with feature engineering and regularization.
- **[price_prediction/lasso.ipynb](price_prediction/lasso.ipynb)**: Lasso regression model for flight price prediction with L1 regularization and feature selection.
- **[price_prediction/xgboost.ipynb](price_prediction/xgboost.ipynb)**: XGBoost gradient boosting model for flight price prediction, capturing non-linear relationships and feature interactions.

## Data Files

- **Kaggle dataset**: dilwong/flightprices (downloaded programmatically in notebooks)
- **[data/sample_20k.csv](data/sample_20k.csv)**: Sample of 20,000 flight itineraries used for model training and testing
- **[data/KG_cost_small.csv](data/KG_cost_small.csv)**: Route‑level panel dataset for market structure analysis and fixed-effects regressions

## Environment

The notebooks are designed for Google Colab, but can run locally with the same Python packages.

Key packages installed within notebooks:
- kaggle, pandas, pyarrow, numpy (data ingestion and manipulation)
- matplotlib, seaborn, plotly (visualization)
- scikit‑learn, xgboost (machine learning modeling)
- geopandas, geodatasets, shapely (geographic mapping)
- linearmodels, statsmodels (panel regressions)

## Detailed Analysis

### 1. Exploratory Data Analysis (Flight_Prices_Analysis.ipynb)

**Data Acquisition:**
- Downloads the Kaggle "dilwong/flightprices" dataset via API
- Converts large CSV to Parquet format for efficient storage and loading
- Stores processed data in Google Drive for persistence across sessions

**Pricing Analysis:**
The notebook explores fare patterns across multiple dimensions:
- **By Airline**: Average base and total fares per carrier
- **Booking Window**: Fare trends vs. days until departure
- **Seat Availability**: Price relationships with remaining seats
- **Airport**: Fare distributions by origin and destination airports
- **Distance**: Fare scaling with total travel distance

**Market Structure Classification (HHI):**
- Filters to single-carrier itineraries
- Computes per-route airline counts and market shares
- Calculates Herfindahl-Hirschman Index (HHI): $HHI = \sum_{i=1}^{n} s_i^2$ where $s_i$ is market share of airline $i$
- Classifies routes:
  - **Monopoly**: HHI > 0.6
  - **Oligopoly**: 0.2 < HHI ≤ 0.6
  - **Competitive**: 0 < HHI ≤ 0.2
- Exports route-level market structure data

### 2. Market Structure & Panel Analysis (Data_science.ipynb)

**Data Preparation:**
1. Loads route-level panel dataset ([KG_cost_small.csv](data/KG_cost_small.csv)) from GitHub
2. Enriches with airport coordinates from OpenFlights database
3. Merges origin and destination lat/lon for each route
4. Filters to U.S. routes only

**2005 Route Snapshot:**
- Creates cross-section of routes for year 2005
- Computes mean HHI per route
- Classifies market types:
  - **Competitive**: HHI ≤ 0.2
  - **Oligopoly**: 0.2 < HHI ≤ 0.6
  - **Monopoly**: HHI > 0.6

**Visualizations:**
- Count plot: Distribution of routes by market type
- Geographic map: U.S. routes colored by market structure using GeoPandas
- City size analysis: Interaction between origin/destination city sizes and market concentration
- Faceted bar charts: Market type percentages across city-size categories

**Panel Fixed-Effects Regressions:**
Uses `linearmodels.PanelOLS` with route and year fixed effects:
1. **Mean fare ~ HHI_route + FE**: Tests if concentration raises average prices
2. **Min fare ~ HHI_route + FE**: Examines concentration impact on lowest fares
3. **Gini ~ HHI_route + FE**: Analyzes whether concentration increases fare inequality
4. **HHI ~ route_city_type + FE**: Studies how city size affects market concentration

All models cluster standard errors at the route level.

### 3. Ridge Regression Price Prediction (ridge.ipynb)

**Data Loading:**
- Loads sample dataset from Google Drive (originally `itineraries_sample.csv`)
- Uses 20k sample for efficient model training

**Market Structure Classification:**
Applies HHI calculation with different thresholds:
- **Monopoly**: HHI > 0.25
- **Oligopoly**: 0.15 < HHI ≤ 0.25
- **Competitive**: HHI ≤ 0.15

**Feature Engineering:**
- **Temporal**: booking_window (days between search and flight, clipped to [0, 365])
- **Route**: num_legs (count from segmented fields), route_distance (winsorized)
- **Capacity**: seatsRemaining (numeric)
- **Categorical**: startingAirport, destinationAirport, segmentsAirlineCode, isNonStop, isRefundable, isBasicEconomy

**Model Pipeline:**
1. **Preprocessing**:
   - Numeric features: Imputation + StandardScaler
   - Categorical features: Imputation + OneHotEncoder
2. **Model**: RidgeCV with cross-validated alpha selection (5-fold)
3. **Train/Test Split**: Temporal split by flightDate (80/20)
4. **Target**: log(totalFare) for better handling of price skewness

**Evaluation Metrics:**
- R² (train and test)
- Median Absolute Error in dollars
- Mean Absolute Percentage Error (MAPE)
- Coefficient analysis (top positive/negative)

### 4. Lasso Regression Price Prediction (lasso.ipynb)

**Data Loading:**
- Uses `sample_20k.csv` from Google Drive
- Implements same HHI-based market classification as ridge model

**Feature Engineering:**
Similar to ridge but with enhanced carrier features:
- **Core features**: booking_window, num_legs, route_distance, seatsRemaining
- **Carrier attributes**: 
  - First/last carrier in itinerary
  - Number of unique carriers
  - Number of carrier changes
  - ULCC flags (NK/F9/G4)
  - LCC flags (WN/B6/AS)
  - Top carrier presence indicators (binary)

**Model Pipeline:**
1. **Preprocessing**: 
   - Similar to ridge but with min_frequency threshold for categorical encoding
   - Reduces dimensionality by grouping rare categories
2. **Model**: LassoCV with cross-validated alpha selection
3. **L1 Regularization**: Automatic feature selection through coefficient shrinkage

**Key Differences from Ridge:**
- L1 penalty drives some coefficients exactly to zero (sparse solutions)
- Better for high-dimensional data with irrelevant features
- Provides implicit feature selection

**Evaluation:**
- Same metrics as ridge (R², MAE, MAPE)
- Feature importance through non-zero coefficients
- Residual analysis and visualization

### 5. XGBoost Price Prediction (xgboost.ipynb)

**Data Loading:**
- Uses local file path for sample dataset
- Loads `itineraries_sample.csv` from local directory

**Why XGBoost?**
- **Non-linear relationships**: Captures complex price dynamics (e.g., U-shaped booking curves)
- **Feature interactions**: Automatically learns how features combine (e.g., LCC × competitive route)
- **Robustness**: Handles missing values and outliers better than linear models
- **Mixed data types**: Native support for categorical variables

**Enhanced Feature Engineering:**

1. **Temporal Features**:
   - lead_time_days (booking window)
   - dep_dayofweek, dep_month, dep_day
   - is_weekend (binary)
   - season (categorical: winter/spring/summer/fall)

2. **Route Features**:
   - route (origin_destination pair)
   - distance (winsorized at 1st and 99th percentiles)
   - route-level statistics from training data

3. **Carrier Features**:
   - Number of segments/legs
   - First/last carrier in itinerary
   - Unique carrier count
   - Carrier changes (connections)
   - ULCC/LCC indicators
   - Top carrier presence flags

4. **Market Structure**:
   - HHI classification (from earlier analysis)
   - Competitive environment indicators

5. **Service Attributes**:
   - isNonStop, isRefundable, isBasicEconomy
   - Cabin class
   - Seats remaining

**Model Configuration:**
- Gradient boosting with tree-based learners
- Hyperparameter tuning (max_depth, learning_rate, n_estimators)
- Early stopping to prevent overfitting
- Feature importance analysis via gain/weight/cover

**Advanced Analysis:**
- SHAP values for model interpretability
- Partial dependence plots for feature effects
- Residual analysis by route and carrier
- Error distribution across market types

**Performance:**
XGBoost typically outperforms linear models (Ridge/Lasso) due to:
- Ability to model non-linearities
- Automatic feature interaction discovery
- Better handling of categorical variables
- Ensemble learning benefits

## Reproducing the Workflow

### Google Colab (Recommended)

1. **Flight_Prices_Analysis.ipynb**:
   - Upload kaggle.json for API access
   - Run download and Parquet conversion cells (one-time setup)
   - Execute analysis cells for aggregations and HHI computation

2. **Data_science.ipynb**:
   - No special setup required (data loaded from GitHub)
   - Run all cells in sequence for panel analysis and visualizations

3. **ridge.ipynb / lasso.ipynb**:
   - Mount Google Drive
   - Ensure `itineraries_sample.csv` or `sample_20k.csv` is in Drive
   - Run cells in order for model training and evaluation

4. **xgboost.ipynb**:
   - For Colab: Update file path to Google Drive location
   - For local: Use local file path (already configured)
   - Run all cells for enhanced feature engineering and XGBoost training

### Local Execution

1. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
   pip install geopandas geodatasets shapely
   pip install linearmodels statsmodels
   pip install pyarrow kaggle
   ```

2. **Update file paths**:
   - Replace Google Drive paths with local directories
   - Skip `drive.mount()` cells
   - Ensure data files are in correct locations

3. **Run notebooks**:
   - Use Jupyter Notebook or JupyterLab
   - Execute cells sequentially
   - Some visualizations may require interactive backends

## Key Findings & Insights

### Market Structure
- Most U.S. routes exhibit oligopolistic structure (0.2 < HHI ≤ 0.6)
- Monopolies more common on smaller city-pair routes
- Competitive markets cluster around major hubs

### Price Determinants
- **Booking window**: U-shaped curve (expensive when last-minute or very early)
- **Seats remaining**: Negative correlation with price (scarcity premium)
- **Route distance**: Strong positive relationship
- **Carrier type**: ULCCs consistently cheaper than legacy carriers
- **Market structure**: Higher HHI associated with elevated fares

### Model Performance
- **Ridge/Lasso**: R² ~ 0.6-0.7, good baseline with interpretability
- **XGBoost**: R² ~ 0.75-0.85, superior predictive power
- Feature importance: booking_window, route_distance, and carrier identity dominate

### Panel Regression Results
- Higher HHI significantly raises mean and minimum fares
- Concentration increases fare inequality (Gini coefficient)
- City size negatively correlates with HHI (more competition at large airports)

## Notes

- **HHI Thresholds**: Vary by notebook due to different analytical purposes:
  - `Flight_Prices_Analysis.ipynb`: 0.6/0.2 cutoffs
  - `ridge.ipynb`/`lasso.ipynb`: 0.25/0.15 cutoffs
  - `Data_science.ipynb`: 0.6/0.2 cutoffs (consistent with economic literature)

- **Data Sources**: 
  - OpenFlights airport database used for geocoding IATA codes
  - Kaggle "dilwong/flightprices" for ticket-level data (83M+ records)
  - Custom route-level panel for temporal analysis

- **Computation**: 
  - Full Kaggle dataset requires significant memory (convert to Parquet)
  - Models trained on 20k sample for efficiency
  - Panel regressions use aggregated route-year observations

## License

See [LICENSE](LICENSE).
