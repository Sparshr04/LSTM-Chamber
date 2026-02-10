# 🌧️ Deep Learning for Rainfall Forecasting: A Comparative LSTM Study
### Case Study: Pune Meteorological Data (2008–2022)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Executive Summary

This project investigates the efficacy of **Long Short-Term Memory (LSTM)** networks for daily rainfall prediction. Using 14 years of historical weather data from Pune, India, we conducted a rigorous architectural search to determine the optimal model depth and configuration for capturing complex monsoon patterns.

**The Verdict:**
Our experiments reveal that a **2-Layer LSTM architecture** provides the optimal balance between model capacity and generalization, achieving an **R² score of 0.501**, significantly outperforming both shallower baselines and deeper, complex networks.

---

## 📊 Interactive Results Dashboard

An interactive HTML dashboard to visualize model performance, efficiency comparisons, and experiment results.

👉 **Live Dashboard:**  
🔗 https://sparshr04.github.io/LSTM-Chamber/

### Highlights
- 📈 Visual comparison of LSTM experiment results
- ⚡ Efficiency & performance metrics
- 🧠 Model behavior insights through interactive charts

> Note: The dashboard is hosted using GitHub Pages for easy public access.

---


## 📊 Key Results at a Glance

We trained and evaluated **7 distinct model configurations**. The comparison below highlights the efficiency of our optimized approach vs. other architectures.

| Rank | Model Architecture | Test R² Score | RMSE | Status |
|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Model-2 (2 Layers)** | **0.501** | **0.046** | **Optimal** |
| 🥈 | Model-1 (1 Layer, High LR) | 0.469 | 0.047 | Baseline |
| 🥉 | Model-5 (5 Layers) | 0.457 | 0.047 | Diminishing Returns |
| ... | ... | ... | ... | ... |
| 7 | Model-6 (6 Layers) | 0.191 | 0.058 | Overfitting/Degradation |

> **📉 Insight:** While deep learning often encourages "going deeper," our results show that for daily weather data, networks deeper than 3 layers suffer from degradation, while single-layer networks struggle to capture complex feature interactions.

---

## 📁 Dataset

**Source:** Pune, India Meteorological Data (2008-2022)
- **Records:** 4,826 daily observations
- **Features:** 
  - Temperature (°C)
  - Humidity (%)
  - Wind Speed (km/h)
  - Wind Direction (degrees)
  - Atmospheric Pressure (hPa)
  - **Target:** Rainfall (mm)

**Preprocessing Pipeline:**
1. **Temporal Aggregation:** Hourly → Daily averages
2. **Missing Value Handling:** Linear interpolation
3. **Normalization:** MinMaxScaler (0-1 range)
4. **Sequence Generation:** 30-day sliding window
5. **Train/Test Split:** 80/20 stratified split

---

## 🏗️ Methodology

### Model Architectures (Table 1)

| Model   | Layers | Learning Rate | Parameters | Training Strategy             |
|---------|--------|---------------|------------|-------------------------------|
| Model-1 | 1      | 0.001         | 69,249     | Baseline                      |
| **Model-2** | **2** | **0.001** | **118,593** | **Best Performer** ⭐        |
| Model-3 | 3      | 0.001         | 130,977    | Moderate Depth                |
| Model-4 | 4      | 0.001         | 134,097    | Deep Network                  |
| Model-5 | 5      | 0.001         | 134,889    | Very Deep                     |
| Model-6 | 6      | 0.001         | 135,093    | Degradation Test              |
| Model-7 | 1      | 0.000001      | 69,249     | Low LR (Paper's Recommendation) |

**Network Configuration:**
- LSTM units per layer: 128
- Activation: ReLU (LSTM default)
- Optimizer: Adam
- Loss Function: Mean Squared Error
- Callbacks: EarlyStopping (patience=10), ModelCheckpoint

### Training Configuration

```python
EPOCHS = 100  # Increased from 20 to allow convergence
BATCH_SIZE = 32
EARLY_STOPPING = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
MODEL_CHECKPOINT = ModelCheckpoint(filepath='saved_models/{model}_best.keras', save_best_only=True)
```

---

## 🎯 Results

### Performance Metrics (Table 2)

| Model   | Layers | Learning Rate | Test RMSE ↓ | Test MSE  | Test MAE | R² Score ↑ | Epochs Trained |
|---------|--------|---------------|-------------|-----------|----------|------------|----------------|
| Model-1 | 1      | 0.001         | 0.047162    | 0.002224  | 0.019648 | **0.469**  | 54             |
| **Model-2** | **2** | **0.001** | **0.045703** | **0.002089** | **0.017756** | **0.501** ⭐ | **64** |
| Model-3 | 3      | 0.001         | 0.048692    | 0.002371  | 0.019462 | **0.434**  | 31             |
| Model-4 | 4      | 0.001         | 0.047711    | 0.002276  | 0.018755 | **0.456**  | 48             |
| Model-5 | 5      | 0.001         | 0.047679    | 0.002273  | 0.018185 | **0.457**  | 47             |
| Model-6 | 6      | 0.001         | 0.058210    | 0.003388  | 0.023834 | **0.191**  | 43             |
| Model-7 | 1      | 0.000001      | 0.056526    | 0.003195  | 0.023580 | **0.237**  | 100            |

### Key Observations

1. **Winner: Model-2 (2 Layers)**
   - Achieved highest R² (0.501) and lowest RMSE (0.046)
   - Converged efficiently in 64 epochs
   - Optimal balance between capacity and generalization

2. **Shallow Network Limitations (Model-7)**
   - Despite 1000-epoch marathon training, R² plateaued at 0.332
   - Very low learning rate (1e-6) caused slow, inefficient learning
   - **15.6x more training time** than Model-2 but worse performance

3. **Deep Network Degradation (Model-6)**
   - 6 layers showed significant performance drop (R²=0.191)
   - Suggests overfitting or optimization challenges
   - Dataset size may not justify such deep architectures

4. **Efficiency Gap**
   - Models 1-5 with LR=0.001 converged in 31-64 epochs
   - Model-7 with LR=1e-6 showed minimal improvement after 700 epochs
   - Standard learning rates proved far more efficient for this problem

### Visualizations

![Performance Gap Comparison](plots/performance_gap_comparison.png)
*Model-2 demonstrated superior efficiency, achieving R² = 0.501 in just 64 epochs, whereas the paper's Model-7 achieved only R² = 0.237 even after 100 epochs (2.1x better performance)*

![Validation Loss Comparison](plots/02_Cost_Function_Comparison.png)
*Convergence speed comparison across all 7 models over 100 epochs*

---

## 🔬 Critical Analysis

### Why Model-2 Outperformed the Paper's Model-7

The original research paper recommended **Model-7 (1 layer, LR=1e-6)** for rainfall prediction. Our experiments on Pune data reveal:

> [!WARNING]
> **Architecture Mismatch:** The paper's recommended configuration may not generalize across different datasets and geographic regions.

**Our Findings:**

1. **Learning Rate Efficiency**
   - LR=1e-6 is too conservative for this dataset
   - Model-7 required 1000+ epochs vs Model-2's 64
   - Validation loss improvements became negligible after epoch 700

2. **Dataset Complexity**
   - Pune's weather patterns may require more representational capacity
   - 2-layer architecture better captures temporal dependencies
   - Single-layer network underfits the data

3. **Training Dynamics**
   - EarlyStopping prevented overfitting (Models 1-6 stopped at 31-64 epochs)
   - Model-7 continued training without significant gains
   - Standard Adam optimization (LR=0.001) proved most effective

**Takeaway:** Hyperparameter tuning must be dataset-specific. Blind replication of research findings can lead to suboptimal results.

---

## 🚀 Installation & Usage

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.13.0
keras>=2.13.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
plotly>=5.14.0
```

### ⚡ Fast Installation (using `uv`)

This project supports [uv](https://github.com/astral-sh/uv) for extremely fast dependency management.

1.  **Install uv:**
    ```bash
    # macOS/Linux
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

    # Windows (PowerShell)
    powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
    
    # Or via pip
    pip install uv
    ```

2.  **Sync dependencies:**
    ```bash
    uv sync
    ```

### Project Structure

```
Weather_Analysis/
├── data/
│   └── pune.csv                        # Raw weather data
├── models/
│   └── saved_models/                   # Best model checkpoints (.keras)
│       ├── Model-1_best.keras
│       ├── Model-2_best.keras
│       └── ...
├── plots/                              # Static visualizations
│   ├── 01_EDA_Analysis.png
│   ├── 02_Cost_Function_Comparison.png
│   ├── 03_Actual_vs_Predicted_Model-2.png
│   └── efficiency_gap_analysis.png
├── marathon_logs/                      # Extended training logs
│   ├── master_log_model7.csv
│   └── history_*.png
├── data_preprocessing.py               # Preprocessing pipeline
├── lstm_model_builder.py               # Model architecture definitions
├── full_experiment.py                  # Main training script
├── generate_dashboard.py               # Interactive visualization
├── experiment_results.csv              # Performance metrics
├── results_dashboard.html              # Interactive dashboard
└── README.md                           # This file
```

---

## 📚 References

1. Original Research Paper: [Insert Citation]
2. Dataset Source: India Meteorological Department
3. LSTM Architecture: Hochreiter & Schmidhuber (1997)
4. Adam Optimizer: Kingma & Ba (2014)

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@Sparshr04](https://github.com/Sparshr04)
- LinkedIn: [Sparsh R](https://www.linkedin.com/in/sparsh-rannaware-7a7a372ba/)

