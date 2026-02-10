"""
Marathon Training Experiment for Model-7
==========================================
Train Model-7 for 1000 epochs with "Bookmark" logging every 100 epochs.
Goal: Determine if Model-7 can eventually beat Model-2's R² of 0.501.

Author: MLOps Engineer & Data Scientist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
import warnings

warnings.filterwarnings("ignore")

# Import our modules
from data_preprocessing import main as preprocess_data
from lstm_model_builder import build_lstm_model

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class BookmarkCallback(Callback):
    """
    Custom callback that logs metrics every 100 epochs (bookmarks).

    At each bookmark:
    - Evaluates on test set
    - Logs RMSE, MAE, R²
    - Saves a snapshot plot of training history
    - Appends to master log DataFrame
    """

    def __init__(
        self,
        X_test,
        y_test,
        model_name,
        bookmark_interval=100,
        output_dir="marathon_logs",
    ):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.bookmark_interval = bookmark_interval
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize master log
        self.master_log = pd.DataFrame(
            columns=[
                "Model",
                "Epoch",
                "Train_Loss",
                "Val_Loss",
                "Test_RMSE",
                "Test_MAE",
                "Test_R2",
            ]
        )

        print(f"\n{'=' * 70}")
        print(f"📊 BOOKMARK CALLBACK INITIALIZED")
        print(f"{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"Bookmark Interval: Every {bookmark_interval} epochs")
        print(f"Output Directory: {output_dir}")
        print(f"{'=' * 70}\n")

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        current_epoch = epoch + 1  # epochs are 0-indexed

        # Check if this is a bookmark epoch
        if current_epoch % self.bookmark_interval == 0:
            print(f"\n{'=' * 70}")
            print(f"📍 BOOKMARK CHECKPOINT - Epoch {current_epoch}")
            print(f"{'=' * 70}")

            # 1. Evaluate on test set
            y_pred = self.model.predict(self.X_test, verbose=0).flatten()

            # 2. Calculate metrics
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            test_mae = mean_absolute_error(self.y_test, y_pred)
            test_r2 = r2_score(self.y_test, y_pred)

            train_loss = logs.get("loss", 0)
            val_loss = logs.get("val_loss", 0)

            print(f"  Train Loss:  {train_loss:.6f}")
            print(f"  Val Loss:    {val_loss:.6f}")
            print(f"  Test RMSE:   {test_rmse:.6f}")
            print(f"  Test MAE:    {test_mae:.6f}")
            print(f"  Test R²:     {test_r2:.6f}")

            # 3. Augment the master log
            new_row = pd.DataFrame(
                [
                    {
                        "Model": self.model_name,
                        "Epoch": current_epoch,
                        "Train_Loss": train_loss,
                        "Val_Loss": val_loss,
                        "Test_RMSE": test_rmse,
                        "Test_MAE": test_mae,
                        "Test_R2": test_r2,
                    }
                ]
            )
            self.master_log = pd.concat([self.master_log, new_row], ignore_index=True)

            # 4. Generate and save snapshot plot
            self._save_snapshot_plot(current_epoch)

            print(f"  ✓ Snapshot saved: history_model7_epoch_{current_epoch}.png")
            print(f"{'=' * 70}\n")

    def _save_snapshot_plot(self, current_epoch):
        """Generate and save a plot of training history up to current epoch."""
        history = self.model.history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Loss curves
        epochs_so_far = range(1, len(history.history["loss"]) + 1)

        ax1.plot(
            epochs_so_far,
            history.history["loss"],
            linewidth=2,
            color="#2E86AB",
            label="Training Loss",
            alpha=0.8,
        )
        ax1.plot(
            epochs_so_far,
            history.history["val_loss"],
            linewidth=2,
            color="#FF6B35",
            label="Validation Loss",
            alpha=0.8,
        )

        ax1.set_title(
            f"Model-7 Training Progress - Epoch {current_epoch}/1000\n"
            + f"Learning Rate: 1e-6",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Loss (MSE)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Add current metrics text box
        latest_train = history.history["loss"][-1]
        latest_val = history.history["val_loss"][-1]
        metrics_text = f"Current Metrics:\n"
        metrics_text += f"Train Loss: {latest_train:.6f}\n"
        metrics_text += f"Val Loss: {latest_val:.6f}"

        ax1.text(
            0.98,
            0.98,
            metrics_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Plot 2: Bookmark metrics progression
        if len(self.master_log) > 0:
            bookmark_epochs = self.master_log["Epoch"].values
            test_r2_values = self.master_log["Test_R2"].values
            test_rmse_values = self.master_log["Test_RMSE"].values

            ax2_twin = ax2.twinx()

            line1 = ax2.plot(
                bookmark_epochs,
                test_rmse_values,
                marker="o",
                linewidth=2.5,
                markersize=8,
                color="#E63946",
                label="Test RMSE",
                alpha=0.8,
            )
            line2 = ax2_twin.plot(
                bookmark_epochs,
                test_r2_values,
                marker="s",
                linewidth=2.5,
                markersize=8,
                color="#06A77D",
                label="Test R²",
                alpha=0.8,
            )

            ax2.set_title(
                "Bookmark Checkpoints - Test Set Performance",
                fontsize=14,
                fontweight="bold",
            )
            ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Test RMSE", fontsize=12, fontweight="bold", color="#E63946")
            ax2_twin.set_ylabel(
                "Test R² Score", fontsize=12, fontweight="bold", color="#06A77D"
            )

            ax2.tick_params(axis="y", labelcolor="#E63946")
            ax2_twin.tick_params(axis="y", labelcolor="#06A77D")

            # Add Model-2's benchmark line
            ax2_twin.axhline(
                y=0.501,
                color="gold",
                linestyle="--",
                linewidth=2,
                label="Model-2 Benchmark (R²=0.501)",
                alpha=0.7,
            )

            # Combine legends
            lines = (
                line1
                + line2
                + [
                    plt.Line2D(
                        [0],
                        [0],
                        color="gold",
                        linestyle="--",
                        linewidth=2,
                        label="Model-2 Benchmark",
                    )
                ]
            )
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc="center right", fontsize=10)

            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/history_model7_epoch_{current_epoch}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def get_master_log(self):
        """Return the master log DataFrame."""
        return self.master_log


def train_model7_marathon():
    """
    Execute the marathon training experiment for Model-7.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MARATHON TRAINING EXPERIMENT - MODEL-7")
    print(" " * 25 + "1000 Epochs Challenge")
    print("=" * 80)

    # Step 1: Load and preprocess data
    print("\n📦 STEP 1: Loading and Preprocessing Data...")
    X_train, y_train, X_test, y_test, scaler = preprocess_data()

    # Step 2: Build Model-7
    print("\n🏗️  STEP 2: Building Model-7...")
    model = build_lstm_model(
        num_layers=1,
        learning_rate=0.000001,
        input_shape=(X_train.shape[1], X_train.shape[2]),
    )

    # Step 3: Setup bookmark callback
    print("\n📊 STEP 3: Setting Up Bookmark Callback...")
    bookmark_callback = BookmarkCallback(
        X_test=X_test,
        y_test=y_test,
        model_name="Model-7",
        bookmark_interval=100,
        output_dir="marathon_logs",
    )

    # Step 4: Train for 1000 epochs
    print("\n🚀 STEP 4: Starting Marathon Training (1000 Epochs)...")
    print("=" * 80)
    print("⏰ This will take a while. Bookmark checkpoints every 100 epochs.")
    print("=" * 80)

    EPOCHS = 1000
    BATCH_SIZE = 32

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[bookmark_callback],
        verbose=1,  # Show progress
    )

    # Step 5: Final evaluation
    print("\n" + "=" * 80)
    print("📊 STEP 5: Final Evaluation")
    print("=" * 80)

    y_pred_final = model.predict(X_test, verbose=0).flatten()
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
    final_mae = mean_absolute_error(y_test, y_pred_final)
    final_r2 = r2_score(y_test, y_pred_final)

    print(f"\n🎯 FINAL RESULTS (Epoch {EPOCHS}):")
    print(f"  Test RMSE: {final_rmse:.6f}")
    print(f"  Test MAE:  {final_mae:.6f}")
    print(f"  Test R²:   {final_r2:.6f}")
    print("\n📊 Model-2 Benchmark:")
    print(f"  Test RMSE: 0.045703")
    print(f"  Test R²:   0.501033")

    # Comparison
    print("\n" + "=" * 80)
    if final_r2 > 0.501033:
        print("🏆 CONCLUSION: Model-7 BEATS Model-2!")
        print(f"   R² improvement: {(final_r2 - 0.501033):.6f}")
    else:
        print("📊 CONCLUSION: Model-2 remains the winner.")
        print(f"   R² gap: {(0.501033 - final_r2):.6f}")
    print("=" * 80)

    # Step 6: Display master log
    master_log = bookmark_callback.get_master_log()
    print("\n" + "=" * 80)
    print("📋 MASTER LOG TABLE - Bookmark Checkpoints")
    print("=" * 80)
    print(master_log.to_string(index=False))

    # Save master log
    master_log.to_csv("marathon_logs/master_log_model7.csv", index=False)
    print("\n✓ Saved: marathon_logs/master_log_model7.csv")

    # Step 7: Generate final comparison plot
    print("\n📈 STEP 6: Generating Final Comparison Plot...")
    generate_final_comparison(master_log, history)

    print("\n" + "=" * 80)
    print(" " * 25 + "✅ MARATHON EXPERIMENT COMPLETE!")
    print("=" * 80)

    return model, history, master_log


def generate_final_comparison(master_log, history):
    """Generate final comparison plot of Model-7's journey."""

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Plot 1: Full training history (1000 epochs)
    epochs = range(1, len(history.history["loss"]) + 1)

    axes[0, 0].plot(
        epochs,
        history.history["loss"],
        linewidth=1.5,
        color="#2E86AB",
        label="Training Loss",
        alpha=0.7,
    )
    axes[0, 0].plot(
        epochs,
        history.history["val_loss"],
        linewidth=1.5,
        color="#FF6B35",
        label="Validation Loss",
        alpha=0.7,
    )
    axes[0, 0].set_title(
        "Model-7: Complete 1000-Epoch Training Journey\nLearning Rate: 1e-6",
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Loss (MSE)", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="upper right", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Bookmark R² progression
    bookmark_epochs = master_log["Epoch"].values
    test_r2_values = master_log["Test_R2"].values

    axes[0, 1].plot(
        bookmark_epochs,
        test_r2_values,
        marker="o",
        linewidth=2.5,
        markersize=10,
        color="#06A77D",
        label="Model-7 Test R²",
        alpha=0.8,
    )
    axes[0, 1].axhline(
        y=0.501033,
        color="gold",
        linestyle="--",
        linewidth=3,
        label="Model-2 Benchmark (R²=0.501)",
        alpha=0.8,
    )
    axes[0, 1].set_title(
        "R² Score Progression - Did Model-7 Beat Model-2?",
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Test R² Score", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="lower right", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Annotate final R²
    final_r2 = test_r2_values[-1]
    axes[0, 1].annotate(
        f"Final R²: {final_r2:.4f}",
        xy=(bookmark_epochs[-1], final_r2),
        xytext=(bookmark_epochs[-1] - 200, final_r2 + 0.05),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
    )

    # Plot 3: RMSE progression
    test_rmse_values = master_log["Test_RMSE"].values

    axes[1, 0].plot(
        bookmark_epochs,
        test_rmse_values,
        marker="s",
        linewidth=2.5,
        markersize=10,
        color="#E63946",
        label="Model-7 Test RMSE",
        alpha=0.8,
    )
    axes[1, 0].axhline(
        y=0.045703,
        color="gold",
        linestyle="--",
        linewidth=3,
        label="Model-2 Benchmark (RMSE=0.046)",
        alpha=0.8,
    )
    axes[1, 0].set_title(
        "RMSE Progression - Lower is Better", fontsize=14, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Test RMSE", fontsize=12, fontweight="bold")
    axes[1, 0].legend(loc="upper right", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Master log table
    axes[1, 1].axis("tight")
    axes[1, 1].axis("off")

    table_data = []
    for _, row in master_log.iterrows():
        table_data.append(
            [
                f"{int(row['Epoch'])}",
                f"{row['Val_Loss']:.6f}",
                f"{row['Test_RMSE']:.6f}",
                f"{row['Test_R2']:.4f}",
            ]
        )

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=["Epoch", "Val Loss", "Test RMSE", "Test R² ↑"],
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.3, 0.3, 0.3],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor("#4CAF50")
        cell.set_text_props(weight="bold", color="white")

    # Highlight best R²
    best_r2_idx = master_log["Test_R2"].idxmax()
    for j in range(4):
        cell = table[(best_r2_idx + 1, j)]
        cell.set_facecolor("#FFF9C4")
        cell.set_text_props(weight="bold")

    axes[1, 1].set_title(
        "Bookmark Checkpoints Summary", fontsize=14, fontweight="bold", pad=20
    )

    plt.suptitle(
        "Model-7 Marathon Training Analysis (1000 Epochs)\n"
        + "Can the Slow Learner Beat the Fast One?",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(
        "marathon_logs/final_comparison_model7_vs_model2.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Saved: marathon_logs/final_comparison_model7_vs_model2.png")
    plt.show()


if __name__ == "__main__":
    model, history, master_log = train_model7_marathon()
