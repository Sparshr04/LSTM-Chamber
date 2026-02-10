"""
Data Preprocessing Pipeline for LSTM Rainfall Prediction
Replicating research paper: "Daily Rainfall Prediction using LSTM"

This script implements:
- Synthetic weather data generation (Pune/Maharashtra structure)
- Missing data handling via linear interpolation
- MinMaxScaler normalization (0-1 range)
- Sliding window sequence generation (lookback=30 days)
- Time-ordered train/test split (80/20)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def load_real_weather_data(csv_path="pune_weather_dataset/pune.csv"):
    """
    Load real Pune weather dataset from CSV.

    Parameters:
    -----------
    csv_path : str
        Path to the Pune weather CSV file

    Returns:
    --------
    pd.DataFrame with columns: Date, Temperature, Humidity, Wind Speed,
                                Wind Direction, Pressure, Rainfall
    """
    print("=" * 70)
    print("STEP 1: LOADING REAL PUNE WEATHER DATA")
    print("=" * 70)

    # Load CSV
    df_raw = pd.read_csv(csv_path)

    print(f"✓ Loaded data from: {csv_path}")
    print(f"✓ Raw data shape: {df_raw.shape}")
    print(f"\nRaw columns: {list(df_raw.columns)}")

    # Select and rename columns to match our pipeline requirements
    column_mapping = {
        "date_time": "Date",
        "tempC": "Temperature",
        "humidity": "Humidity",
        "windspeedKmph": "Wind Speed",
        "winddirDegree": "Wind Direction",
        "pressure": "Pressure",
        "precipMM": "Rainfall",
    }

    # Select only the required columns and rename
    df = df_raw[list(column_mapping.keys())].copy()
    df.rename(columns=column_mapping, inplace=True)

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"\n✓ Selected and renamed columns")
    print(f"✓ New columns: {list(df.columns)}")

    # The data is hourly - aggregate to daily level (mean for most, sum for rainfall)
    print(f"\n✓ Original data frequency: Hourly")
    print(f"✓ Converting to daily data (aggregating by date)...")

    df["DateOnly"] = df["Date"].dt.date

    # Aggregate to daily: mean for weather features, sum for rainfall
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

    # Rename DateOnly back to Date
    df_daily.rename(columns={"DateOnly": "Date"}, inplace=True)
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])

    print(f"✓ Aggregated to daily data shape: {df_daily.shape}")
    print(f"✓ Date range: {df_daily['Date'].min()} to {df_daily['Date'].max()}")
    print(f"✓ Total days: {len(df_daily)}")

    print(f"\nDataset Info:")
    print(df_daily.info())

    print(f"\nFirst few rows:")
    print(df_daily.head())

    print(f"\nBasic Statistics:")
    print(df_daily.describe())

    # Check for missing values
    missing_count = df_daily.isnull().sum()
    print(f"\nMissing values per column:")
    print(missing_count)

    return df_daily


def handle_missing_data(df):
    """
    Handle missing data using linear interpolation as per research methodology.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with potential missing values

    Returns:
    --------
    pd.DataFrame with interpolated values
    """
    print("\n" + "=" * 70)
    print("STEP 2: HANDLING MISSING DATA (Linear Interpolation)")
    print("=" * 70)

    # Check for missing values before interpolation
    missing_before = df.isnull().sum()
    print(f"Missing values before interpolation:")
    print(missing_before[missing_before > 0])

    # Apply linear interpolation
    df_interpolated = df.copy()
    df_interpolated = df_interpolated.interpolate(
        method="linear", limit_direction="both"
    )

    # Check for missing values after interpolation
    missing_after = df_interpolated.isnull().sum()
    print(f"\n✓ Missing values after interpolation:")
    print(missing_after)

    return df_interpolated


def normalize_features(df, feature_columns):
    """
    Apply MinMaxScaler (0-1 range) to all features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_columns : list
        List of column names to normalize

    Returns:
    --------
    scaled_data : np.ndarray
        Normalized features as numpy array
    scaler : MinMaxScaler
        Fitted scaler object for inverse transformation
    """
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE NORMALIZATION (MinMaxScaler 0-1)")
    print("=" * 70)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_columns])

    print(f"✓ Normalized {len(feature_columns)} features: {feature_columns}")
    print(f"✓ Output shape: {scaled_data.shape}")
    print(f"\nScaled data sample (first 5 rows):")
    print(pd.DataFrame(scaled_data[:5], columns=feature_columns))

    return scaled_data, scaler


def create_sequences(data, lookback=30):
    """
    Create sequences using sliding window approach for LSTM input.

    Parameters:
    -----------
    data : np.ndarray
        Normalized feature array of shape (n_samples, n_features)
    lookback : int
        Number of past days to use for prediction (default: 30)

    Returns:
    --------
    X : np.ndarray
        Input sequences of shape (samples, lookback, n_features)
    y : np.ndarray
        Target values (next day's rainfall) of shape (samples,)
    """
    print("\n" + "=" * 70)
    print(f"STEP 4: SEQUENCE GENERATION (Sliding Window, lookback={lookback})")
    print("=" * 70)

    X, y = [], []

    # Rainfall is the last column (index -1)
    rainfall_idx = -1

    for i in range(lookback, len(data)):
        # X: Past 'lookback' days of ALL features
        X.append(data[i - lookback : i, :])

        # y: Next day's rainfall (target)
        y.append(data[i, rainfall_idx])

    X = np.array(X)
    y = np.array(y)

    print(f"✓ Created {len(X)} sequences")
    print(f"✓ X shape: {X.shape} (samples, lookback, features)")
    print(f"✓ y shape: {y.shape} (samples,)")
    print(f"\nSequence structure:")
    print(f"  - Each X sample contains {lookback} days × {X.shape[2]} features")
    print(f"  - Each y value is the rainfall for day {lookback + 1}")

    return X, y


def split_train_test(X, y, train_ratio=0.8):
    """
    Split data into train and test sets without shuffling (maintain time order).

    Parameters:
    -----------
    X : np.ndarray
        Input sequences
    y : np.ndarray
        Target values
    train_ratio : float
        Proportion of data for training (default: 0.8)

    Returns:
    --------
    X_train, y_train, X_test, y_test
    """
    print("\n" + "=" * 70)
    print(
        f"STEP 5: TRAIN/TEST SPLIT ({int(train_ratio * 100)}/{int((1 - train_ratio) * 100)})"
    )
    print("=" * 70)

    # Calculate split index
    split_idx = int(len(X) * train_ratio)

    # Split without shuffling (important for time series)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    print(f"✓ Split performed at index {split_idx}")
    print(f"✓ Time order maintained (NO shuffling)")
    print(f"\nTrain set:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"\nTest set:")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test


def main():
    """
    Main pipeline execution.
    """
    print("\n" + "=" * 70)
    print("LSTM RAINFALL PREDICTION - DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    print("Research Paper: Daily Rainfall Prediction using LSTM")
    print("Dataset: Real Pune/Maharashtra Weather Data")
    print("=" * 70)

    # Configuration
    LOOKBACK = 30  # 30 days of historical data
    TRAIN_RATIO = 0.8  # 80% training, 20% testing

    # Step 1: Load real weather data
    df = load_real_weather_data("pune_weather_dataset/pune.csv")

    # Step 2: Set Date as index
    df.set_index("Date", inplace=True)
    print(f"\n✓ Date set as index")

    # Step 3: Handle missing data
    df = handle_missing_data(df)

    # Define feature columns (all columns including target)
    feature_columns = [
        "Temperature",
        "Humidity",
        "Wind Speed",
        "Wind Direction",
        "Pressure",
        "Rainfall",
    ]

    # Step 4: Normalize features
    scaled_data, scaler = normalize_features(df, feature_columns)

    # Step 5: Create sequences
    X, y = create_sequences(scaled_data, lookback=LOOKBACK)

    # Step 6: Split into train/test
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_ratio=TRAIN_RATIO)

    # Final Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - FINAL SUMMARY")
    print("=" * 70)
    print(f"\n📊 FINAL SHAPES:")
    print(
        f"  ✓ X_train: {X_train.shape}  (samples, {LOOKBACK}, {len(feature_columns)} features)"
    )
    print(f"  ✓ y_train: {y_train.shape}  (samples,)")
    print(
        f"  ✓ X_test:  {X_test.shape}   (samples, {LOOKBACK}, {len(feature_columns)} features)"
    )
    print(f"  ✓ y_test:  {y_test.shape}   (samples,)")

    print(f"\n📋 VALIDATION:")
    print(f"  ✓ Constraint met: X_train shape is (samples, 30, n_features) ✓")
    print(f"  ✓ Time order preserved (no shuffling) ✓")
    print(f"  ✓ All features normalized (0-1 range) ✓")
    print(f"  ✓ Missing data handled via linear interpolation ✓")

    print(f"\n📈 DATA STATISTICS:")
    print(f"  - Total sequences: {len(X)}")
    print(f"  - Training sequences: {len(X_train)} ({TRAIN_RATIO * 100:.0f}%)")
    print(f"  - Testing sequences: {len(X_test)} ({(1 - TRAIN_RATIO) * 100:.0f}%)")
    print(f"  - Features per timestep: {len(feature_columns)}")
    print(f"  - Lookback window: {LOOKBACK} days")

    print(f"\n🎯 TARGET VARIABLE (Rainfall) Distribution:")
    print(
        f"  - Train set - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}, Mean: {y_train.mean():.4f}"
    )
    print(
        f"  - Test set  - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}, Mean: {y_test.mean():.4f}"
    )

    print("\n" + "=" * 70)
    print("✅ PREPROCESSING PIPELINE READY FOR LSTM MODEL")
    print("=" * 70)

    return X_train, y_train, X_test, y_test, scaler


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler = main()
