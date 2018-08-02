# regression

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

boston_housing = keras.datasets.boston_housing;

(training_set, training_labels), (test_set, test_labels) = boston_housing.load_data();

# Shuffle the training training set
order = np.argsort(np.random.random(training_labels.shape));
training_set = training_set[order];
training_labels = training_labels[order];

print("Training set: {}".format(training_set.shape));
print("Testing set: {}".format(test_set.shape));

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT'];
df = pd.DataFrame(training_set, columns=column_names);
print(df.head())

# Normalize features
mean = training_set.mean(axis=0);
std = training_set.std(axis=0);
training_set = (training_set - mean)/std;
test_set = (test_set - mean)/std;



def build_model():

    model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(training_set.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
    ]);

    optimizer = tf.train.RMSPropOptimizer(0.001);

    model.compile(loss='mse',
                   optimizer=optimizer,
                   metrics=['mae']);
    return model;

model = build_model();
model.summary();

class printDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('');
        print('.', end='');

epochs = 500;

# Store training stats
history = model.fit(training_set, training_labels, epochs=epochs,
                    validation_split=0.2, verbose=0,
                    callbacks=[printDot()]);


def plot_history(history):
    plt.figure();
    plt.xlabel('Epoch');
    plt.ylabel('Mean Abs Error');
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss');
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label = 'Val loss');
    plt.legend();
    plt.ylim([0,5]);
    plt.show();

model = build_model()

# The patience parameter is the amount of epochs to check for improvement.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20);

history = model.fit(training_set, training_labels, epochs=epochs,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, printDot()]);

plot_history(history);
[loss, mae] = model.evaluate(test_set, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:7.2f}".format(mae * 1000))

# Predict
test_predictions = model.predict(test_set).flatten()
print(test_predictions)
