import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

# Load preprocessed data
train_df = pd.read_csv('https://raw.githubusercontent.com/amansharma-tech/PECF/main/train_df.csv')
df1 = pd.read_csv('https://raw.githubusercontent.com/amansharma-tech/PECF/main/df1.csv')
X_test = np.load('https://github.com/amansharma-tech/PECF/raw/main/X_test.npy')
y_test = np.load('https://github.com/amansharma-tech/PECF/raw/main/y_test.npy')

# Load trained linear model
linear_model = load_model('https://github.com/amansharma-tech/PECF/raw/blob/main/linear_model.h5')

# Load test results
results = pd.read_csv('https://raw.githubusercontent.com/amansharma-tech/PECF/blob/main/test_results.csv')

# Calculate evaluation metrics
mse = mean_squared_error(results['y_test_actual'].tail(48), results['linear_model_pred'].tail(48))
mae = mean_absolute_error(results['y_test_actual'].tail(48), results['linear_model_pred'].tail(48))
rmse = np.sqrt(mse)
r_squared = r2_score(results['y_test_actual'].tail(48), results['linear_model_pred'].tail(48))

# Plot actuals vs predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results['y_test_actual'].tail(48), color='b', marker='s')
ax.plot(results['linear_model_pred'].tail(48), color='darkorange', marker='X')
ax.set_title(f"Actuals vs Linear Model Predictions (Last 48 hours)\nRMSE: {rmse:.2f}, MSE: {mse:.2f}, MAE: {mae:.2f}, R-squared: {r_squared:.2f}")
ax.set_xlabel("Time")
ax.set_ylabel("Load")
ax.legend(["Actuals", "Predictions"])
st.pyplot(fig)

# Display evaluation metrics
st.write(f'Linear Model Test MSE: {mse:.2f}')
st.write(f'Linear Model Test MAE: {mae:.2f}')
st.write(f'Linear Model Test RMSE: {rmse:.2f}')
st.write(f'Linear Model Test R-squared: {r_squared:.2f}')
