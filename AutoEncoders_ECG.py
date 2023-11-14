import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings

# add your file path here
CSV_PATH = ""

warnings.filterwarnings('ignore')

df = pd.read_csv(CSV_PATH)
data = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 21)

#Now lets Normalize the data
#First we will calculate the maximum and minimum value from the training set
min = tf.reduce_min(train_data)
max = tf.reduce_max(train_data)

# #Now we will use the formula (data - min)/(max - min)
train_data = (train_data - min)/(max - min)
test_data = (test_data - min)/(max - min)

#Converting the data into float
train_data = tf.cast(train_data, dtype=tf.float32)
test_data = tf.cast(test_data, dtype=tf.float32)

#The labels are either 0 or 1, so I will convert them into boolean(true or false)
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

#Now let's separate the data for normal ECG from that of abnormal ones
#Normal ECG data
n_train_data = train_data[train_labels]
n_test_data = test_data[test_labels]

#Abnormal ECG data
an_train_data = train_data[~train_labels]
an_test_data = test_data[~test_labels]

# print(n_train_data)

#Lets plot a normal ECG
plt.plot(np.arange(162), n_train_data[0], 'g')
plt.plot(np.arange(162), an_train_data[0], 'r')
plt.grid()
plt.title('Normal ECG(Green) / Abnormal ECG(Red)')
plt.show()

# #Lets plot one from abnormal ECG
# plt.plot(np.arange(162), an_train_data[0])
# plt.grid()
# plt.title('Abnormal ECG')
# plt.show()


class detector(Model):
  def __init__(self):
    super(detector, self).__init__()
    self.encoder = tf.keras.Sequential([
                                        layers.Dense(32, activation='relu'),
                                        layers.Dense(16, activation='relu'),
                                        layers.Dense(8, activation='relu')
    ])
    self.decoder = tf.keras.Sequential([
                                        layers.Dense(16, activation='relu'),
                                        layers.Dense(32, activation='relu'),
                                        layers.Dense(162, activation='sigmoid')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

#Let's compile and train the model!!
autoencoder = detector()
autoencoder.compile(optimizer='adam', loss='mae')
autoencoder.fit(n_train_data, n_train_data, epochs = 20, batch_size=5, validation_data=(n_test_data, n_test_data))

#Now let's define a function in order to plot the original ECG and reconstructed ones and also show the error
def plot(data, n):
  enc_img = autoencoder.encoder(data)
  dec_img = autoencoder.decoder(enc_img)
  plt.plot(data[n], 'b')
  plt.plot(dec_img[n], 'r')
  plt.fill_between(np.arange(162), data[n], dec_img[n], color = 'lightcoral')
  plt.legend(labels=['Input', 'Reconstruction', 'Error'])
  plt.show()

plot(n_test_data, 0)
plot(an_test_data, 0)


reconstructed = autoencoder(n_train_data)
train_loss = losses.mae(reconstructed, n_train_data)
t = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", t)
plt.hist(train_loss, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.title('Reconstruction error Train')
plt.show()

reconstructions = autoencoder.predict(an_test_data)
test_loss = tf.keras.losses.mae(reconstructions, an_test_data)

plt.hist(test_loss, bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.title('Reconstruction error Anomaly Test')
plt.show()

def prediction(model, data, threshold):
  rec = model(data)
  loss = losses.mae(rec, data)
  return tf.math.less(loss, threshold)

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(preds, labels):
  print("Accuracy = {}".format(accuracy_score(labels, preds)))
  print("Precision = {}".format(precision_score(labels, preds)))
  print("Recall = {}".format(recall_score(labels, preds)))
  print("F1 Score = {}".format(f1_score(labels, preds)))



preds = predict(autoencoder, test_data, t)
print_stats(preds, test_labels)
# print(pred)

#Lets see some more result visually !!
# plot(n_test_data, 0)
# plot(n_test_data, 1)
# plot(n_test_data, 3)


# df2 = pd.read_csv(CSV_PATH)
# test1 = df2.iloc[:,:-1].values
# min = tf.reduce_min(test1)
# max = tf.reduce_max(test1)
# test1 = (test1 - min)/(max - min)
# train_data1 = tf.cast(test1, dtype=tf.float32)
# #Lets plot a normal ECG
# plt.plot(np.arange(162), train_data1[2])
# plt.grid()
# plt.title('Normal ECG')
# plt.show()
#
# plot(train_data1, 1)
# plot(train_data1, 2)