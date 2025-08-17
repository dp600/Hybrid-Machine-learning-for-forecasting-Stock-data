---
title: Hybrid ML for Next-Day Stock Forecasting (AAPL)
description: XGBoost → LSTM with chronological split
---

# Hybrid Machine Learning for Next-Day Stock Forecasting (AAPL)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1XT7KS9PuJnqa8g7dV8AorjTCziD_3icp?usp=sharing)

**Design.** XGBoost predicts next-day log return; an LSTM consumes the ordered XGBoost predictions to produce the final forecast.  
**Split.** Strict **chronological** train/test.  
**Explainability.** **SHAP** on the XGBoost stage.

- 👉 **Full README:** [View on GitHub](https://github.com/dp600/Hybrid-Machine-learning-for-forecasting-Stock-data#readme)
- 📓 **Colab (view-only):** link above
