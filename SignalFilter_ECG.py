import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.signal import find_peaks, butter, filtfilt
import csv


url_string = "2014_10_01-08_42_43"
number_string = "007"

csv_path = "archive_1/diabetes_subset_ecg_data/diabetes_subset_ecg_data/{0}/sensor_data/{1}/{1}_ECG.csv".format(number_string,url_string )


def signal_filter(df_data, sample_range):
    output_list = []
    # Preprocessing
    fs = 250  # Sampling rate
    lowcut = 3  # Lower frequency of bandpass filter
    highcut = 45  # Higher frequency of bandpass filter

    # Convert ECG signal to mV
    ecg_signal = (df_data['EcgWaveform'].values.astype(float) - 1024) / 200

    # Apply bandpass filter
    nyquist_rate = fs / 2.0
    low = lowcut / nyquist_rate
    high = highcut / nyquist_rate
    b, a = butter(2, [low, high], btype='band')
    ecg_cleaned = filtfilt(b, a, ecg_signal)

    # Find R-peaks
    r_peaks, _ = find_peaks(ecg_cleaned, height=0.5, distance=fs * 0.3)

    # Select R-peaks within the specified sample range
    r_peaks_in_range = r_peaks[((r_peaks - int(0.2 * fs)) > sample_range[0]) & ((r_peaks + int(0.45 * fs))< sample_range[1])]

    # Find QRS complex
    qrs_start = r_peaks_in_range - int(0.1 * fs)
    qrs_end = r_peaks_in_range + int(0.3 * fs)
    qrs_peaks = [np.argmax(ecg_cleaned[s:e]) + s for s, e in zip(qrs_start, qrs_end)]

    # Find QT interval
    qt_start = [qrs_peak - int(0.2 * fs) for qrs_peak in qrs_peaks]
    qt_end = [r_peak + int(0.45 * fs) for r_peak in r_peaks_in_range]
    # qt_intervals = [(e - s) / fs for s, e in zip(qt_start, qt_end)]
    if len(qt_start) > 0  and len(qt_end) > 0:
        ecg_list = list(ecg_cleaned[qt_start[0]:qt_end[0]])
        print(len(ecg_list))
        if "diabetes" in csv_path:
            ecg_list.append(0)

        else:
            ecg_list.append(1)
        return ecg_list
    else:
        return []


ecg_data1 = pd.read_csv(csv_path)
output_data = []
min = ecg_data1[60000:60250]['EcgWaveform'].idxmin()
for i in range(150):

    # output_data[str(i)] = ecg_data1['EcgWaveform'][min:min+500].to_numpy()
    # output_data.reset_index(drop=True)
    # min = min+500
    ans_data = signal_filter(ecg_data1[min:min+500], [0,500])
    if len(ans_data) > 160:
        output_data.append(ans_data)
    min = min + 500

b = output_data
print(b)
file_name = "./data/007_d_4.csv"

with open(file_name,"w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv)
    csvWriter.writerows(output_data)
