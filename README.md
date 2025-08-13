# Hybrid ML for Next-Day Stock Forecasting (AAPL)
*A fixed **XGBoost → LSTM** pipeline with a **chronological split** to forecast next-day Apple closing price.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](#)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XT7KS9PuJnqa8g7dV8AorjTCziD_3icp?usp=sharing)
[![Based in Ireland](https://img.shields.io/badge/Based%20in-Ireland-169B62)](#)

## Executive Summary
- **Design:** Train **XGBoost** to predict **t+1 log return**; feed the ordered XGBoost predictions into an **LSTM** to produce the **final forecast** (no compromise on order).
- **Discipline:** **Chronological split** (past → future), strict target alignment, no leakage.
- **Explainability:** **SHAP** on the XGBoost stage (global + local).
- **Deliverable:** Reproducible pipeline with **RMSE / MAE** and **Directional Accuracy** on a held-out test period.

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
1) Feature engineering on OHLCV  
2) **XGBoost** predicts next-day log return  
3) **SHAP** explains XGBoost  
4) **LSTM** consumes the **ordered XGBoost prediction stream** to model temporal dependence  
5) Reconstruct next-day price from predicted return  
6) Evaluate with RMSE/MAE (returns & prices) and Directional Accuracy

## How to Run
- Use the Colab notebook (badge above) **or** run locally with your own environment and config.  
- Stages: **prepare_data → build_features → train_xgb → shap_report → make_sequences → train_lstm → evaluate**.

## Config (excerpt)
- `configs/config.yaml` holds dates, features, model params, and the **chronological cutoff**.
- Example settings:  
  - `data.start: 1995-01-01` · `data.end: 2025-08-01` · `data.cutoff_train: 2018-12-31`  
  - `seq.lookback: 15` · `xgb.n_estimators: 100` · `xgb.learning_rate: 0.03`

## Results  *(replace with your numbers)*
| Split | RMSE (return) | MAE (return) | Directional Accuracy |
|------:|---------------:|-------------:|---------------------:|
| Test  | YOUR_RMSE      | YOUR_MAE     | YOUR_HIT_RATE%       |

**Figures**  
`reports/figures/xgb_shap_summary.png` · `reports/figures/pred_vs_actual.png`

## Reproducibility & Quality
- Deterministic seeds (42); no data leakage (fit transforms on train only).
- Config-driven runs; artifacts saved for audit; concise, documented functions.

## Ethics & Disclaimer
Research purpose only; **not financial advice**. Verify data licensing and corporate actions.

## Contact (Ireland)
**Abdullah Al Tawab — Dublin** · Open to walk-throughs and technical discussion.






 
