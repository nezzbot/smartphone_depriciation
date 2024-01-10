import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import time
import pandas as pd
import csv
import os
from dataPhone import data1, data2, data3



df = pd.DataFrame.from_dict(data1)

# Define the Holt-Winters Exponential Smoothing function
def holt_winters_exponential_smoothing(t, alpha, beta, gamma, phi, initial_level, initial_trend, *initial_seasonal):
    level = initial_level
    trend = initial_trend
    seasonal = np.array(initial_seasonal)
    predictions = []

    for i in range(len(t)):
        forecast = level + phi * trend + seasonal[i % len(seasonal)]
        predictions.append(forecast)

        # Update level, trend, and seasonal components
        level = alpha * (t[i] - forecast) + (1 - alpha) * level
        trend = beta * (level - initial_level) + (1 - beta) * trend
        seasonal[i % len(seasonal)] = gamma * (t[i] - forecast) + (1 - gamma) * seasonal[i % len(seasonal)]

    return np.array(predictions)


# Define the logistic growth function
def logistic_growth(t, L, k, t0, b):
    return L / (1 + b * np.exp(-k * (t - t0)))


# Define the Exponential Decay function
def exponential_decay(t, A, B):
    return A * np.exp(-B * t) 


time_real = np.array([i for i in range(25)])
# print(time_real)

PhoneType = []
PhoneValues_real = []
PhoneValues_Scaled = []
HWD_pred = []
HWD_MSE = []
HWD_MeanTime = []
ExpD_pred = []
ExpD_MSE = []
ExpD_MeanTime = []
LogD_pred = []
LogD_MSE = []
LogD_MeanTime = []

for cols in df.columns:
    PhoneType.append(cols)

    # Prepare historical price data
    gadget_value_real = np.array(df[cols])
    print(gadget_value_real)
    gadget_value_real_scaled = gadget_value_real / gadget_value_real[0]

    PhoneValues_real.append(gadget_value_real[1:])
    PhoneValues_Scaled.append(gadget_value_real_scaled[1:])

    # Holt-Winter Damped
    # Initial parameter guesses (replace with reasonable estimates)
    initial_guess = [0.5, 0.5, 0.5, 0.5, gadget_value_real_scaled[1], 0, 0.0, 0.0, 0.0, 0.0]

    time_container=[]
    i=0
    while(i<5):
    # Measure the time taken for model fitting
        start_time = time.time()

        # Fit the Holt-Winters Exponential Smoothing model to the data
        parameters, covariance = curve_fit(holt_winters_exponential_smoothing, time_real[1:], gadget_value_real_scaled[1:], p0=initial_guess, maxfev=10000000)

        # End time
        end_time = time.time()
        processing_time = end_time - start_time
        time_container.append(processing_time)
        # print(f"Time taken for model fitting: {processing_time:.6f} seconds")
        i+=1

    mean_time=(time_container[0]+time_container[1]+time_container[2]+time_container[3]
            +time_container[4])/5
    print(f"mean time : {mean_time}")

    # Generate predictions using the fitted parameters
    predictions = holt_winters_exponential_smoothing(time_real[1:], *parameters)

    # Calculate accuracy metrics
    # r2 = r2_score(gadget_value_real[1:], predictions)
    mse = mean_squared_error(gadget_value_real_scaled[1:], predictions)

    # print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    HWD_MSE.append(mse)
    HWD_MeanTime.append(mean_time)
    HWD_pred.append(predictions)


    # Logistics Decay
    # Initial parameter guesses (replace with reasonable estimates)
    initial_guess = [max(gadget_value_real_scaled[1:]), 0.1, np.median(time_real[1:]), min(gadget_value_real_scaled[1:])]
    
    # Measure the time taken for model fitting
    time_container=[]
    i=0
    while(i<5):
        start_time = time.time()

        # Fit the logistic growth model to the data
        parameters, covariance = curve_fit(logistic_growth, time_real[1:], gadget_value_real_scaled[1:], p0=initial_guess, maxfev=5000)
        # print(f"parameters : {parameters}")
        # End time
        end_time = time.time()
        processing_time = end_time - start_time
        time_container.append(processing_time)
        # print(f"Time taken for model fitting: {processing_time:.6f} seconds")
        i+=1

    mean_time=(time_container[0]+time_container[1]+time_container[2]+time_container[3]
            +time_container[4])/5
    print(f"mean time : {mean_time}")

    # Generate predictions using the fitted parameters
    predictions = logistic_growth(time_real[1:], *parameters)

    # Calculate accuracy metrics
    # r2 = r2_score(gadget_value_real[1:], predictions)
    mse = mean_squared_error(gadget_value_real_scaled[1:], predictions)

    # print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    LogD_MSE.append(mse)
    LogD_MeanTime.append(mean_time)
    LogD_pred.append(predictions)


    # Exponential Decay
    # Initial parameter guesses (replace with reasonable estimates)
    initial_guess = [max(gadget_value_real_scaled[1:]), 0.1 ] 
    # min(gadget_value_real)

    time_container=[]
    i=0
    while(i<5):
        start_time = time.time()

        # Fit the Exponential Decay model to the data
        parameters, covariance = curve_fit(exponential_decay, time_real[1:], gadget_value_real_scaled[1:], p0=initial_guess)
        # print(f"parameters :")
        # End time
        end_time = time.time()
        processing_time = end_time - start_time
        # print(f"Time taken for Exponential Decay Model Fitting: {processing_time:.6f} seconds")
        # Measure the time taken for model fitting
        time_container.append(processing_time)
        i+=1

    # print(f"waktu : {time_container}")
    mean_time=(time_container[0]+time_container[1]+time_container[2]+time_container[3]
            +time_container[4])/5
    print(f"mean time : {mean_time}")


    # Generate predictions using the fitted parameters
    predictions = exponential_decay(time_real[1:], *parameters)

    # Calculate accuracy metrics
    r2 = r2_score(gadget_value_real_scaled[1:], predictions)
    mse = mean_squared_error(gadget_value_real_scaled[1:], predictions)

    # print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    ExpD_MSE.append(mse)
    ExpD_MeanTime.append(mean_time)
    ExpD_pred.append(predictions)

print(np.shape(PhoneType))
print(np.shape(HWD_MSE))
print(np.shape(LogD_MSE))
print(np.shape(ExpD_MSE))

data_MSE = pd.DataFrame.from_dict({
    "Jenis"      : PhoneType,
    "HoltWintersDamped"    : HWD_MSE,
    "LogisticDecay"        : LogD_MSE,
    "ExponentialDecay"     : ExpD_MSE
})

data_MeanTime = pd.DataFrame.from_dict({
    "Jenis"      : PhoneType,
    "HoltWintersDamped"    : HWD_MeanTime,
    "LogisticDecay"        : LogD_MeanTime,
    "ExponentialDecay"     : ExpD_MeanTime
})

# data_plot_dict = {}
# for i in range(len(PhoneType)):
#     data_plot_dict[PhoneType[i]] = {
#         "Real Price"    : PhoneValues_real[i],
#         "Scaled Price"  : PhoneValues_Scaled[i],
#         "Predict HWD"   : HWD_pred[i],
#         "Predict LogD"  : LogD_pred[i],
#         "Predict ExpD"  : ExpD_pred[i]
#     }

data_plot_dict = {}
for i in range(len(PhoneType)):
    data_plot_dict[PhoneType[i]] = {
        "Real Price"    : PhoneValues_real[i],
        "Scaled Price"  : PhoneValues_Scaled[i],
        "Predict HWD"   : HWD_pred[i],
        "Predict LogD"  : LogD_pred[i],
        "Predict ExpD"  : ExpD_pred[i]
    }


data_plot = pd.DataFrame.from_dict(data_plot_dict)
# data_plot.to_csv('prediction_results.csv', mode='a', index=False, header=not os.path.exists('prediction_results.csv'))
# data_MSE.to_csv('mse_results.csv', mode='a', index=False, header=not os.path.exists('mse_results.csv'))
# data_MeanTime.to_csv('mean_time_results.csv', mode='a', index=False, header=not os.path.exists('mean_time_results.csv'))


phone_to_plot = "Huawei P10"

if phone_to_plot in data_plot.columns:
    scaled_price = data_plot[phone_to_plot]["Scaled Price"]
    predict_hwd = data_plot[phone_to_plot]["Predict HWD"]
    predict_logd = data_plot[phone_to_plot]["Predict LogD"]
    predict_expd = data_plot[phone_to_plot]["Predict ExpD"]

    # Check if all arrays have the same length
    if len(scaled_price) == len(predict_hwd) == len(predict_logd) == len(predict_expd):
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

        plt.plot(scaled_price, label='Harga aktual (scaled)', marker='o', linestyle='-', color='blue')
        plt.plot(predict_hwd, label='Holt-Winter damped', marker='o', linestyle='--', color='red')
        plt.plot(predict_logd, label='logistics decay', marker='o', linestyle='-.', color='orange')
        plt.plot(predict_expd, label='exponential decay', marker='o', linestyle=':', color='green')

        plt.title(f'Prediksi vs Harga Aktual untuk {phone_to_plot}')
        plt.xlabel('Bulan')
        plt.ylabel('Harga Scaled')
        plt.legend()
        plt.grid(True)
        plt.savefig("hasil_prediksiHuaweiP10.png")
        plt.show()
    else:
        print("Error: Lengths of arrays are not the same.")
else:
    print(f"Error: {phone_to_plot} not found in the dataset.")

# plt.plot(data_plot["Samsung 8"]["Scaled Price"], label='Data Asli Terskala')
# plt.plot(data_plot["Samsung 8"]["Predict HWD"], label='Holt-Winter Damped')
# plt.plot(data_plot["Samsung 8"]["Predict LogD"], label='Logistics Decay')
# plt.plot(data_plot["Samsung 8"]["Predict ExpD"], label='Exponential Decay')

# plt.show()