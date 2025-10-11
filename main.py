import tensorflow as tf
from keras.api.models import load_model
from keras.api.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import joblib

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


time_step = 15

# Load test data
closedf = np.load("closedf.npy", allow_pickle=True)
close_stock = np.load("close_stock.npy", allow_pickle=True)
tf_X_test = np.load("tf_X_test.npy", allow_pickle=True)
tf_y_test = np.load("tf_y_test.npy", allow_pickle=True)
sk_X_test = np.load("sk_X_test.npy", allow_pickle=True)
sk_y_test = np.load("sk_y_test.npy", allow_pickle=True)

sk_X_test = sk_X_test.reshape(sk_X_test.shape[0], -1)

scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf))



#Linear regression
model_linear = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(time_step,28)),  # Input layer for 6 features
    tf.keras.layers.Dense(units=50, activation=None),  # First hidden layer
    tf.keras.layers.Dense(units=50, activation=None),  # Second hidden layer
    tf.keras.layers.Dense(units=50, activation=None),  # Third hidden layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1, activation=None)   # Output layer
])
model_linear.compile(optimizer= 'adam',loss="mse")  # MSE is the loss function for regression

#Random forest
model_random = RandomForestRegressor(n_estimators=100, random_state=42)

#LSTM
model_lstm = tf.keras.Sequential()
model_lstm.add(tf.keras.layers.LSTM(50,return_sequences = True,input_shape = (time_step,28)))
model_lstm.add(tf.keras.layers.LSTM(50,return_sequences = True))
model_lstm.add(tf.keras.layers.LSTM(50))
model_lstm.add(tf.keras.layers.Dropout(0.20))
model_lstm.add(tf.keras.layers.Dense(1))
model_lstm.compile(loss = "mean_squared_error",optimizer = 'adam')

#Load Model
model_linear = load_model("linear_regression/model/model_feature.h5", custom_objects={"mse": MeanSquaredError()})
model_lstm = load_model("lstm/model/model_feature.h5", custom_objects={"mean_squared_error": MeanSquaredError()})
model_random = joblib.load("random_forest/model/model_feature.pkl")

# Make predictions
test_predict_linear = model_linear.predict(tf_X_test)
print("1")
test_predict_random = model_random.predict(sk_X_test)
print("2")
test_predict_lstm = model_lstm.predict(tf_X_test)
print("3")

print(test_predict_linear.shape, test_predict_random.shape, test_predict_lstm.shape)

test_predict_random = np.reshape(test_predict_random, (-1,1))
print(test_predict_random.shape)


# Inverse transform predictions

# linear regression
test_predict_expanded_linear = np.zeros((test_predict_linear.shape[0], 28))
test_predict_expanded_linear[:, 0] = test_predict_linear[:, 0]
test_predict_linear = scaler.inverse_transform(test_predict_expanded_linear)[:, 0].reshape(-1, 1)
original_ytest_linear = scaler.inverse_transform(np.hstack([tf_y_test.reshape(-1, 1), np.zeros((tf_y_test.shape[0], 27))]))[:, 0]

# random forest
test_predict_expanded_random = np.zeros((test_predict_random.shape[0], 28))
test_predict_expanded_random[:, 0] = test_predict_random[:, 0]
test_predict_random = scaler.inverse_transform(test_predict_expanded_random)[:, 0].reshape(-1, 1)
original_ytest_random = scaler.inverse_transform(np.hstack([sk_y_test.reshape(-1, 1), np.zeros((sk_y_test.shape[0], 27))]))[:, 0]

# lstm
test_predict_expanded_lstm = np.zeros((test_predict_lstm.shape[0], 28))
test_predict_expanded_lstm[:, 0] = test_predict_lstm[:, 0]
test_predict_lstm = scaler.inverse_transform(test_predict_expanded_lstm)[:, 0].reshape(-1, 1)
original_ytest_lstm = scaler.inverse_transform(np.hstack([tf_y_test.reshape(-1, 1), np.zeros((tf_y_test.shape[0], 27))]))[:, 0]

import matplotlib.pyplot as plt
def plot_model_comparisons(
    stock_df, stock_prices, test_predict_linear, test_predict_random, test_predict_lstm, time_step, title="Model Comparisons"
):
    """
    Plots multiple subplots for different models comparing actual vs. predicted prices.
    """

    # Align dates to match prediction lengths
    aligned_dates_linear_lstm = stock_prices['date'].iloc[-len(test_predict_linear):].values
    aligned_dates_random = stock_prices['date'].iloc[-len(test_predict_random):].values
    actual_prices_linear_lstm = stock_prices['close'].iloc[-len(test_predict_linear):].values
    actual_prices_random = stock_prices['close'].iloc[-len(test_predict_random):].values

    # Initialize subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Add Linear Regression plot
    axes[0].plot(
        aligned_dates_linear_lstm,
        actual_prices_linear_lstm,
        label="Actual Price",
        color="blue",
    )
    axes[0].plot(
        aligned_dates_linear_lstm,
        test_predict_linear.flatten(),
        label="Linear Predicted",
        color="red",
    )
    axes[0].set_title("Linear Regression")
    axes[0].legend()

    # Add Random Forest plot
    axes[1].plot(
        aligned_dates_random,
        actual_prices_random,
        label="Actual Price",
        color="blue",
    )
    axes[1].plot(
        aligned_dates_random,
        test_predict_random.flatten(),
        label="Random Forest Predicted",
        color="green",
    )
    axes[1].set_title("Random Forest")
    axes[1].legend()

    # Add LSTM plot
    axes[2].plot(
        aligned_dates_linear_lstm,
        actual_prices_linear_lstm,
        label="Actual Price",
        color="blue",
    )
    axes[2].plot(
        aligned_dates_linear_lstm,
        test_predict_lstm.flatten(),
        label="LSTM Predicted",
        color="purple",
    )
    axes[2].set_title("LSTM")
    axes[2].legend()

    # Format x-axis
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    axes[-1].set_xlabel("Date")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Example stock_prices as a pandas DataFrame

stock_prices = pd.DataFrame(close_stock[:, :2], columns=['date', 'close'])

plot_model_comparisons(
    stock_df=closedf,
    stock_prices=stock_prices,
    test_predict_linear=test_predict_linear,
    test_predict_random=test_predict_random,
    test_predict_lstm=test_predict_lstm,
    time_step=time_step
)
