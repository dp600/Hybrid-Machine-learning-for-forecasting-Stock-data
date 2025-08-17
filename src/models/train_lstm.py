"""
train_lstm.py — Train LSTM on the ordered XGB prediction stream (ŷ).
Builds sequences (lookback L), trains, and saves the model + validation arrays.
"""

import os
import random
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# Set seeds for reproducibility
SEED= 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
df=pd.read_csv('/content/apple_feature_engineered_cleaned.csv', parse_dates=['Date'])
df.dropna(inplace=True)
#Add log return
df['Log_Return']= np.log(df['Close']/df['Close'].shift(1))
df.dropna(inplace=True)

# Define Features and Target
features=['RSI','MACD','SMA_10','EMA_10','Momentum','Volatility','BB_lower', 'BB_range','Lag_1','Lag_2','Lag_3']
x= df[features]
y= df['Log_Return']
scaler=StandardScaler()
X_scaled= scaler.fit_transform(x)
X_train, X_test, y_train,y_test = train_test_split(X_scaled,y,shuffle=False, test_size=0.2)
# XGBoost
xgb_model=XGBRegressor(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    random_state=42
)
xgb_model.fit(X_scaled,y)
xgb_preds_full=xgb_model.predict(X_scaled)
# LSTM on XGBOOST Output
def create_sequences(data, window_size=15):
  X_seq, y_seq= [],[]
  for i in range(len(data) -window_size):
    X_seq.append(data[i:i+window_size])
    y_seq.append(data[i+ window_size])
  return np.array(X_seq), np.array(y_seq)
X_lstm, y_lstm= create_sequences(xgb_preds_full, window_size=15)
X_lstm=X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1],1))

split=int(len(X_lstm)*0.8)
X_train, X_test = X_lstm[:split], X_lstm[split:]
y_train, y_test = y_lstm[:split], y_lstm[split:]

#define Model
lstm_model= tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1],1)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=40, batch_size=32,validation_split=0.1)

#Predict and reconstruct
y_pred=lstm_model.predict(X_test)
start_price=df['Close'].iloc[len(df)-len(y_test)-1]

recon_pred=[start_price]
for r in y_pred.flatten():
  recon_pred.append(recon_pred[-1] * np.exp(r))
recon_pred=recon_pred[1:]

recon_actual=[start_price]
for r in y_test:
  recon_actual.append(recon_actual[-1]*np.exp(r))
recon_actual= recon_actual[1:]
#Evaluate
rmse=np.sqrt(mean_squared_error(recon_actual, recon_pred))
r2=r2_score(recon_actual, recon_pred)
print(f'RMSE: {rmse}')
print(f'R²: {r2}')

