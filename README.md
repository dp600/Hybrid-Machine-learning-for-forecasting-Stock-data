# Hybrid ML for Next-Day Stock Forecasting (APPLE)
*A fixed **XGBoost → LSTM** pipeline with a **chronological split** to forecast next-day Apple closing price.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Cite this repo](https://img.shields.io/badge/Cite-CITATION.cff-blue)](CITATION.cff)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XT7KS9PuJnqa8g7dV8AorjTCziD_3icp?usp=sharing)
[![Based in Ireland](https://img.shields.io/badge/Based%20in-Ireland-169B62)](#)

## Executive Summary
- **Design:** Train **XGBoost** to predict **t+1 log return**; feed the ordered XGBoost predictions into an **LSTM** to produce the **final forecast** (no compromise on order).
- **Discipline:** **Chronological split** (past → future), strict target alignment, no leakage.
- **Explainability:** **SHAP** on the XGBoost stage (global + local).
- **Deliverable:** Reproducible pipeline with **RMSE / R^2** and **Directional Accuracy** on a held-out test period.

## Problem & Target
- **Question:** Can a sequential hybrid (**XGBoost → LSTM**) outperform standalone models for next-day AAPL?
- **Target:** Next-day log return of close; final price is reconstructed from the predicted return.

## Data & Features
- ~30 years of AAPL daily OHLCV.
- Engineered features include: RSI, MACD, SMA_10, EMA_10, Momentum, Volatility, Bollinger Bands (lower, range), and Lag_1–Lag_3.

## Train / Validation / Test
- **Split:** Chronological. Rows on/before the cutoff date form train/validation; rows after the cutoff form test.
- **Leakage controls:** Any scaling/encoding is fit on **train only**; the target is strictly **t+1**.

## Pipeline (high level)

1) Feature engineering on OHLCV (Open, High, Low, Close, Volume) 
2) **XGBoost** predicts next-day log return and then reconstruct price
3) **SHAP** explains feature contribution to XGBOOST  
4) **LSTM** consumes the **ordered XGBoost prediction stream** to model temporal dependence  
5) Then reconstruct next-day price from predicted log return  
6) Evaluate with RMSE/R^2 (returns & prices) and Directional Accuracy

## How to Run
- Use the Colab notebook (badge above) **or** run locally with your own environment and config.  
- Stages: **prepare_data → build_features → train_xgb → shap_report → make_sequences → train_lstm → evaluate**.

- 
## Results  

<img width="149" height="35" alt="image" src="https://github.com/user-attachments/assets/d70343b2-9bae-4e7c-aa79-1b5ef03ef3e9" />
      

**Figures of SHAP analysis and Final output**  
<img width="610" height="440" alt="image" src="https://github.com/user-attachments/assets/ef718a34-acd3-4d78-8f34-1475bb847e75" />

<img width="1079" height="436" alt="image" src="https://github.com/user-attachments/assets/d1595006-a134-4ce1-9ed5-4c3c73919094" />


## Reproducibility & Quality
- Deterministic seeds (42); no data leakage (fit transforms on train only).
- Config-driven runs; artifacts saved for audit; concise, documented functions.

## Project files (quick access)

- [`configs/config.yaml`](configs/config.yaml)
- [`src/data/download_data.py`](src/data/download_data.py)
- [`src/models/train_xgb.py`](src/models/train_xgb.py)
- [`src/models/train_lstm.py`](src/models/train_lstm.py)

  
## Ethics & Disclaimer
Research purpose only; **not financial advice**. Verify data licensing and corporate actions.

## Contact (Ireland)
**Abdullah Al Tawab — Dublin** · Open to walk-throughs and technical discussion.







 
