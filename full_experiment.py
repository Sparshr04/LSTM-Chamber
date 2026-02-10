"""
LSTM Rainfall Prediction - Full Experiment Execution
=====================================================

This script executes the complete experiment:
1. EDA - Correlation heatmap and rainfall time series
2. Train all 7 model configurations from Table 1
3. Evaluate performance metrics (RMSE, MSE, MAE, R²)
4. Generate comparison visualizations
5. Display performance tables

Author: Senior Data Scientist
Research: Daily Rainfall Prediction using LSTM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
import os

warnings.filterwarnings("ignore")

# Import our modules
from data_preprocessing import main as preprocess_data
from lstm_model_builder import build_lstm_model

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# ============================================================================
# PART 1: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================


def perform_eda(df_original):
    """
    Generate EDA visualizations:
    1. Correlation heatmap
    2. Rainfall time series over 13 years

    Parameters:
    -----------
    df_original : pd.DataFrame
        Original daily weather data with Date index
    """
    print("\n" + "=" * 70)
    print("PART 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # ========== PLOT 1: Correlation Heatmap ==========
    print("\n1. Generating Correlation Heatmap...")

    correlation_matrix = df_original.corr()

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=axes[0],
    )

    axes[0].set_title(
        "Feature Correlation Heatmap - Pune Weather Data (2008-2022)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Highlight Rainfall correlations
    print("\n   Key Correlations with Rainfall:")
    rainfall_corr = correlation_matrix["Rainfall"].sort_values(ascending=False)
    for feature, corr in rainfall_corr.items():
        if feature != "Rainfall":
            print(f"   • {feature:<15}: {corr:>7.4f}")

    # ========== PLOT 2: Rainfall Time Series ==========
    print("\n2. Generating Rainfall Time Series Plot...")

    axes[1].plot(
        df_original.index,
        df_original["Rainfall"],
        linewidth=0.8,
        alpha=0.7,
        color="steelblue",
        label="Daily Rainfall",
    )

    # Add 30-day moving average to show trends
    df_original["Rainfall_MA30"] = df_original["Rainfall"].rolling(window=30).mean()
    axes[1].plot(
        df_original.index,
        df_original["Rainfall_MA30"],
        linewidth=2,
        color="darkred",
        alpha=0.8,
        label="30-Day Moving Average",
    )

    axes[1].set_title(
        "Daily Rainfall Pattern - Pune (2008-2022)\nShowing Monsoon Seasonality",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    axes[1].set_xlabel("Year", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Rainfall (mm)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Add statistics box
    stats_text = f"Total Days: {len(df_original)}\n"
    stats_text += f"Mean Rainfall: {df_original['Rainfall'].mean():.2f} mm\n"
    stats_text += f"Max Rainfall: {df_original['Rainfall'].max():.2f} mm\n"
    stats_text += f"Days with Rain: {(df_original['Rainfall'] > 0).sum()} ({(df_original['Rainfall'] > 0).sum() / len(df_original) * 100:.1f}%)"

    axes[1].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("01_EDA_Analysis.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved: 01_EDA_Analysis.png")
    plt.show()

    return correlation_matrix


# ============================================================================
# PART 2: EXPERIMENT EXECUTION - Train All 7 Models
# ============================================================================


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all 7 model configurations from Table 1.

    Returns:
    --------
    results_df : pd.DataFrame
        Performance metrics for each model
    histories : dict
        Training history for each model
    models : dict
        Trained model objects
    """
    print("\n" + "=" * 70)
    print("PART 2: TRAINING ALL MODEL CONFIGURATIONS")
    print("=" * 70)

    # Define all 7 configurations from Table 1
    configurations = [
        {"model": "Model-1", "layers": 1, "lr": 0.001},
        {"model": "Model-2", "layers": 2, "lr": 0.001},
        {"model": "Model-3", "layers": 3, "lr": 0.001},
        {"model": "Model-4", "layers": 4, "lr": 0.001},
        {"model": "Model-5", "layers": 5, "lr": 0.001},
        {"model": "Model-6", "layers": 6, "lr": 0.001},
        {"model": "Model-7", "layers": 1, "lr": 0.000001},
    ]

    models = {}
    histories = {}
    results = []

    EPOCHS = 100  # Increased from 20 to allow Model-7 to converge
    BATCH_SIZE = 32

    # Create directory for saving best models
    os.makedirs("saved_models", exist_ok=True)

    for config in configurations:
        print(f"\n{'=' * 70}")
        print(
            f"Training {config['model']}: {config['layers']} layers, LR={config['lr']}"
        )
        print(f"{'=' * 70}")

        # Build model
        model = build_lstm_model(
            num_layers=config["layers"],
            learning_rate=config["lr"],
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )

        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )

        model_checkpoint = ModelCheckpoint(
            filepath=f"saved_models/{config['model']}_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )

        # Train model
        print(f"\nTraining for up to {EPOCHS} epochs with batch size {BATCH_SIZE}...")
        print(f"Using EarlyStopping (patience=10) and ModelCheckpoint...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, model_checkpoint],
            verbose=0,  # Silent training, we'll show progress summary
        )

        # Predictions
        y_pred_train = model.predict(X_train, verbose=0).flatten()
        y_pred_test = model.predict(X_test, verbose=0).flatten()

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        # Store results
        results.append(
            {
                "Model": config["model"],
                "Layers": config["layers"],
                "Learning Rate": config["lr"],
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
                "Test MSE": test_mse,
                "Test MAE": test_mae,
                "R² Score": test_r2,
                "Parameters": model.count_params(),
            }
        )

        models[config["model"]] = model
        histories[config["model"]] = history

        # Print summary
        print(f"\n✓ Training Complete!")
        print(f"  Epochs Trained:   {len(history.history['loss'])}")
        print(f"  Final Train Loss: {history.history['loss'][-1]:.6f}")
        print(f"  Final Val Loss:   {history.history['val_loss'][-1]:.6f}")
        print(f"  Best Val Loss:    {min(history.history['val_loss']):.6f}")
        print(f"  Test RMSE:        {test_rmse:.6f}")
        print(f"  Test R² Score:    {test_r2:.6f}")

    results_df = pd.DataFrame(results)

    return results_df, histories, models


# ============================================================================
# PART 3: VISUALIZATIONS
# ============================================================================


def plot_cost_function_comparison(histories):
    """
    Plot validation loss curves for all 7 models.
    """
    print("\n" + "=" * 70)
    print("PART 3: GENERATING COST FUNCTION COMPARISON")
    print("=" * 70)

    plt.figure(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (model_name, history) in enumerate(histories.items()):
        plt.plot(
            history.history["val_loss"],
            label=model_name,
            linewidth=2.5,
            color=colors[idx],
            alpha=0.8,
        )

    plt.title(
        "Validation Loss Comparison - All 7 LSTM Configurations\nPune Rainfall Prediction (2008-2022)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Validation Loss (MSE)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("02_Cost_Function_Comparison.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 02_Cost_Function_Comparison.png")
    plt.show()


def plot_actual_vs_predicted(
    y_test, model, X_test, model_name="Best Model", num_days=100
):
    """
    Plot actual vs predicted rainfall for the first N days of test data.
    """
    print("\n" + "=" * 70)
    print(f"PART 4: GENERATING ACTUAL VS PREDICTED ({model_name})")
    print("=" * 70)

    # Get predictions
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Plot first num_days
    days = range(1, num_days + 1)

    plt.figure(figsize=(14, 7))

    plt.plot(
        days,
        y_test[:num_days],
        linewidth=2.5,
        color="#2E86AB",
        label="Actual Rainfall",
        marker="o",
        markersize=4,
        alpha=0.7,
    )

    plt.plot(
        days,
        y_pred[:num_days],
        linewidth=2.5,
        color="#FF6B35",
        label="Predicted Rainfall",
        marker="s",
        markersize=4,
        alpha=0.7,
    )

    plt.title(
        f"Rainfall Prediction Performance - {model_name}\nFirst {num_days} Days of Test Data (Normalized Scale)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Day", fontsize=12, fontweight="bold")
    plt.ylabel("Rainfall (Normalized 0-1)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Calculate metrics for these days
    rmse = np.sqrt(mean_squared_error(y_test[:num_days], y_pred[:num_days]))
    mae = mean_absolute_error(y_test[:num_days], y_pred[:num_days])
    r2 = r2_score(y_test[:num_days], y_pred[:num_days])

    # Add metrics box
    metrics_text = f"Metrics (First {num_days} Days):\n"
    metrics_text += f"RMSE: {rmse:.4f}\n"
    metrics_text += f"MAE:  {mae:.4f}\n"
    metrics_text += f"R²:   {r2:.4f}"

    plt.text(
        0.02,
        0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    plt.savefig(
        f"03_Actual_vs_Predicted_{model_name.replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"✓ Saved: 03_Actual_vs_Predicted_{model_name.replace(' ', '_')}.png")
    plt.show()


# ============================================================================
# PART 4: RESULTS TABLES
# ============================================================================


def display_results_tables(results_df):
    """
    Display configuration and performance tables.
    """
    print("\n" + "=" * 70)
    print("FINAL RESULTS - PERFORMANCE COMPARISON TABLES")
    print("=" * 70)

    # Table 1: Configuration
    print("\n📋 TABLE 1: Model Configurations")
    print("-" * 70)
    config_table = results_df[["Model", "Layers", "Learning Rate", "Parameters"]].copy()
    print(config_table.to_string(index=False))

    # Table 2: Performance Metrics
    print("\n\n📊 TABLE 2: Performance Metrics")
    print("-" * 70)
    perf_table = results_df[
        ["Model", "Test RMSE", "Test MSE", "Test MAE", "R² Score"]
    ].copy()
    print(perf_table.to_string(index=False))

    # Highlight best model
    best_model_idx = results_df["Test RMSE"].idxmin()
    best_model = results_df.iloc[best_model_idx]

    print("\n" + "=" * 70)
    print("🏆 BEST MODEL")
    print("=" * 70)
    print(f"Model:       {best_model['Model']}")
    print(f"Layers:      {best_model['Layers']}")
    print(f"Learning Rate: {best_model['Learning Rate']}")
    print(f"Test RMSE:   {best_model['Test RMSE']:.6f}")
    print(f"Test R² Score: {best_model['R² Score']:.6f}")
    print("=" * 70)

    return best_model["Model"]


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """
    Execute the full experimental pipeline.
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "LSTM RAINFALL PREDICTION - FULL EXPERIMENT")
    print(" " * 20 + "Research Paper Replication")
    print("=" * 80)

    # Step 1: Load and preprocess data
    print("\n📦 STEP 1: Loading and Preprocessing Real Pune Weather Data...")
    X_train, y_train, X_test, y_test, scaler = preprocess_data()

    # Load original data for EDA
    from data_preprocessing import load_real_weather_data

    df_original = load_real_weather_data("pune_weather_dataset/pune.csv")
    df_original.set_index("Date", inplace=True)

    # Step 2: EDA
    print("\n📊 STEP 2: Performing Exploratory Data Analysis...")
    correlation_matrix = perform_eda(df_original)

    # Step 3: Train all models
    print("\n🚀 STEP 3: Training All 7 Model Configurations...")
    results_df, histories, models = train_all_models(X_train, y_train, X_test, y_test)

    # Step 4: Visualizations
    print("\n📈 STEP 4: Generating Visualizations...")
    plot_cost_function_comparison(histories)

    # Step 5: Display tables
    print("\n📋 STEP 5: Displaying Results Tables...")
    best_model_name = display_results_tables(results_df)

    # Step 6: Plot best model predictions
    print(f"\n🎯 STEP 6: Visualizing Best Model ({best_model_name}) Predictions...")
    plot_actual_vs_predicted(
        y_test, models[best_model_name], X_test, model_name=best_model_name
    )

    # Save results to CSV
    results_df.to_csv("experiment_results.csv", index=False)
    print("\n✓ Saved results to: experiment_results.csv")

    print("\n" + "=" * 80)
    print(" " * 25 + "✅ EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  1. 01_EDA_Analysis.png")
    print("  2. 02_Cost_Function_Comparison.png")
    print(f"  3. 03_Actual_vs_Predicted_{best_model_name.replace(' ', '_')}.png")
    print("  4. experiment_results.csv")
    print("=" * 80)

    return results_df, histories, models


if __name__ == "__main__":
    results_df, histories, models = main()
