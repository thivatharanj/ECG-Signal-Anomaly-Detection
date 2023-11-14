from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


csv_path = ""


df = pd.read_csv(csv_path)
# signal = df[60000:61000]['EcgWaveform']
# plt.figure(figsize = (15, 7))
# plt.plot(signal)
# plt.title('ECG Signal')
# plt.grid(True)
# plt.show()
#
# signal = pd.Series(signal)


def moving_average(series, n):
    # Calculate average of last n observations
    return np.average(series[-n:])


def plotMovingAverage(series, pandas_df, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies

    """

    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=pandas_df.index, columns=pandas_df['EcgWaveform'])
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

plotMovingAverage(df[60000:61000]['EcgWaveform'], df[60000:61000], 3, plot_anomalies=False, plot_intervals=True)
# plotMovingAverage(df[0:1600]['EcgWaveform'], df[0:1600], 2, plot_anomalies=False, plot_intervals=True)