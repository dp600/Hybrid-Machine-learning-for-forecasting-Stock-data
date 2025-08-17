"""
train_xgb.py — Train XGBoost to predict next-day log return (y_{t+1}).
Chronological split only. Saves model + validation arrays for later steps.
"""
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df= pd.read_csv('/content/apple_feature_engineered_cleaned.csv', parse_dates=['Date'])
#Create Target: log Return
df['Log_Return']=np.log(df['Close']/df['Close'].shift(1))
df.dropna(inplace=True)
# Define Features and Target
selected_features=['RSI','MACD','SMA_10','EMA_10','Momentum','Volatility','BB_lower', 'BB_range','Lag_1','Lag_2','Lag_3']
x= df[selected_features]
y= df['Log_Return']
#Normalize Features
scaler=StandardScaler()
X_scaled= scaler.fit_transform(x)
#Chronological Train/Test Split
X_train, X_test, y_train,y_test = train_test_split(X_scaled,y,shuffle=False, test_size=0.2)

#Train XGBoost
model=XGBRegressor(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    random_state=42
)
model.fit(X_train, y_train)
#Predict log return
y_pred_log_return=model.predict(X_test)
#Evaluate
rmse=np.sqrt(mean_squared_error(y_test, y_pred_log_return))
r2=r2_score(y_test, y_pred_log_return)
print(f'RMSE: {rmse}')
print(f'R²: {r2}')

#Reconstruct predicted prices (if needed)
reconstruct_price= [df['Close'].iloc[-len(y_test)-1]] #Start fromlast known close

for r in y_pred_log_return:
  next_price = reconstruct_price[-1] * np.exp(r)
  reconstruct_price.append(next_price)
reconstruct_price=reconstruct_price[1:] #Drop initial

#Evaluate Reconstructed price
actual_prices= df['Close'].iloc[-len(y_test):].values
rmse_actual= np.sqrt(mean_squared_error(actual_prices, reconstruct_price))
r2_actual= r2_score(actual_prices, reconstruct_price)
print(f'RMSE(Actual price):{rmse_actual}')
print(f'R² (Actual price):{r2_actual}')
