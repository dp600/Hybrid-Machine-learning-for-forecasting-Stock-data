
{% include_relative README.md %}

## Data acquisition (auto-synced)

{% include_relative src/data/download_data.py %}


## Train XGBoost (auto-synced)

{% include_relative src/models/train_xgb.py %}


## Train LSTM (auto-synced)
{% include_relative src/models/train_lstm.py %}
