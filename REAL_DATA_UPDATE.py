"""
Updated Data Loading Code for Real Pune Weather Dataset
=========================================================

This shows the key changes made to data_preprocessing.py to load real data
instead of synthetic data.
"""

# ============================================================================
# UPDATED FUNCTION: load_real_weather_data()
# ============================================================================


def load_real_weather_data(csv_path="pune_weather_dataset/pune.csv"):
    """
    Load real Pune weather dataset from CSV.

    Steps:
    1. Load CSV file with 25 columns
    2. Map original columns to required names:
       - date_time      → Date
       - tempC          → Temperature
       - humidity       → Humidity
       - windspeedKmph  → Wind Speed
       - winddirDegree  → Wind Direction
       - pressure       → Pressure
       - precipMM       → Rainfall
    3. Convert hourly data to daily aggregates (mean for features, sum for rainfall)
    """

    # Load CSV
    df_raw = pd.read_csv(csv_path)

    # Column mapping
    column_mapping = {
        "date_time": "Date",
        "tempC": "Temperature",
        "humidity": "Humidity",
        "windspeedKmph": "Wind Speed",
        "winddirDegree": "Wind Direction",
        "pressure": "Pressure",
        "precipMM": "Rainfall",
    }

    # Select and rename columns
    df = df_raw[list(column_mapping.keys())].copy()
    df.rename(columns=column_mapping, inplace=True)

    # Convert to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Aggregate hourly to daily data
    df["DateOnly"] = df["Date"].dt.date
    df_daily = (
        df.groupby("DateOnly")
        .agg(
            {
                "Temperature": "mean",
                "Humidity": "mean",
                "Wind Speed": "mean",
                "Wind Direction": "mean",
                "Pressure": "mean",
                "Rainfall": "sum",  # Total daily rainfall
            }
        )
        .reset_index()
    )

    df_daily.rename(columns={"DateOnly": "Date"}, inplace=True)
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])

    return df_daily


# ============================================================================
# UPDATED MAIN FUNCTION
# ============================================================================


def main():
    # Old: df = generate_synthetic_weather_data(start_date='2020-01-01', days=1000)
    # NEW:
    df = load_real_weather_data("pune_weather_dataset/pune.csv")

    # Rest of the pipeline remains the same
    df.set_index("Date", inplace=True)
    df = handle_missing_data(df)
    scaled_data, scaler = normalize_features(df, feature_columns)
    X, y = create_sequences(scaled_data, lookback=30)
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_ratio=0.8)


# ============================================================================
# RESULTS WITH REAL DATA
# ============================================================================

"""
Dataset Statistics:
- Raw hourly data: 116,136 rows (2008-2018, ~10 years)
- Daily aggregated: 4,839 days
- Date range: 2008-12-11 to 2022-03-11
- No missing values after aggregation

Final Shapes (REAL DATA):
✓ X_train: (3847, 30, 6)  ← 3847 sequences vs 776 with synthetic data
✓ y_train: (3847,)
✓ X_test:  (962, 30, 6)   ← 962 sequences vs 194 with synthetic data
✓ y_test:  (962,)

Comparison:
                    Synthetic    Real Data    Increase
Total sequences:    970          4809         +395%
Training samples:   776          3847         +395%
Testing samples:    194          962          +395%

Rainfall Distribution (Normalized 0-1):
- Train: Min=0.0000, Max=0.7929, Mean=0.0201
- Test:  Min=0.0000, Max=1.0000, Mean=0.0258

Key Differences:
1. Much larger dataset (4809 vs 970 sequences)
2. Real monsoon patterns from actual weather data
3. Higher variability in rainfall (max 263.6mm in original data)
4. No artificial missing values - real data is complete
"""
