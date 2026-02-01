# 🎾 Mitsui Commodity Price Prediction Engine

<img width="800"  alt="image" src="https://github.com/user-attachments/assets/b979a315-45b6-4d9d-a431-1e0805eb2f9d" />

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Complete-green)]()

A scalable Deep Learning system for high-dimensional time-series forecasting, developed for the [Mitsui Bussan Commodities Kaggle Competition](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge).

## 📌 Project Overview

This project tackles the challenge of predicting price returns for **424 distinct financial targets** (commodities, equities, and differentials) across varying holding periods (1–4 days). The solution required handling complex temporal dependencies, market regime changes, and high-volume feature engineering.

**Key Achievement:** Engineered a decoupled, modular pipeline capable of training and serializing 400+ independent LSTM models, optimized for GPU acceleration.

---

## 🏗 System Architecture

To handle the computational load of 424 targets, I moved away from a monolithic script and designed a **3-stage modular pipeline**:

### Stage 1: Distributed Feature Engineering
*   **Input:** Raw OHLCV data for 106 underlying assets.
*   **Process:** Generated 1000+ features per target, including Lagged Returns, Rolling Statistics (Mean/Std/Min/Max), and Technical Indicators (RSI, MACD, Volatility).
*   **Optimization:** Decoupled feature generation from training to allow for intensive compute without blocking GPU resources.

### Stage 2: Feature Selection & Model Factory
*   **Filtration:** Applied a rigorous 2-step selection process to reduce noise:
    1.  **Mutual Information:** Top 300 features selected.
    2.  **Random Forest Regressor:** Top 100 features finalized based on Gini importance.
*   **Training:** Iterative training loop using **Keras/TensorFlow**.
*   **Architecture:** Individual Multi-Layer LSTM networks calibrated for each target.
*   **Output:** 424 serialized `.keras` model artifacts ready for inference.

### Stage 3: Inference Engine
*   **Design:** A lightweight inference script that loads pre-trained model artifacts dynamically.
*   **Logic:** Standard Scaling applied in real-time using fitted scalers from Stage 1.

---

## 🛠 Methodology

*   **Data Preprocessing:** Forward-fill imputation for missing values to preserve temporal order; standard scaling for neural network stability.
*   **Model Validation:** Time-series split (strictly causal) to prevent look-ahead bias.
*   **Performance:** Achieved Sharpe Ratios > 1.5 on multiple validation sets, demonstrating strong signal capture in volatile market conditions.

## 🚀 Key Learnings

1.  **MLOps Implementation:** Moving from "notebook code" to "modular scripts" was essential for managing hundreds of models.
2.  **Deep Learning for Finance:** LSTM networks proved highly effective at capturing non-linear temporal dependencies, provided they are trained for sufficient epochs (20+) to escape local minima.
3.  **Engineering Rigor:** The importance of early end-to-end pipeline testing. Building a robust submission API is as critical as the model accuracy itself.

## 💻 Tech Stack

*   **Core:** Python, Pandas, NumPy
*   **Deep Learning:** Keras, TensorFlow (LSTM)
*   **Machine Learning:** Scikit-Learn (Random Forest, Mutual Information)
*   **Hardware:** Multi-GPU Acceleration

## 📂 Project Structure

```text
├── notebooks/                  # Modular Jupyter notebooks (Feature Eng -> Train -> Inference)
├── models/                     # Serialized Keras models (.h5)
├── requirements.txt            # Dependencies
└── README.md                   # Documentation

