# basic-classification

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__);

# Loading Fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist;
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data();
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];

# Data Preprocess
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.gca().grid(False)

# We scale these values to a range of 0 to 1 before feeding to the neural network model
train_images = train_images / 255.0;
test_images = test_images / 255.0;

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid('off')
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])


# Model building
# Setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
    keras.layers.Dense(128, activation=tf.nn.relu), # Neural layer, 128 nodes (neurons)
    keras.layers.Dense(10, activation=tf.nn.softmax)# Neural layer, 10 nodes (neurons), softmax
])

# Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),       # Optimizer - This is how the model is updated based on the data it sees and its loss function
              loss='sparse_categorical_crossentropy',   # Loss functions - This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
              metrics=['accuracy'])                     # Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

# Train the model
model.fit(train_images, train_labels, epochs = 5);


# Evaluate accuracy against test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid('off')
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     predicted_label = np.argmax(predictions[i])
#     true_label = test_labels[i]
#     if predicted_label == true_label:
#       color = 'green'
#     else:
#       color = 'red'
#     plt.xlabel("{} ({})".format(class_names[predicted_label],
#                                   class_names[true_label]),
#                                   color=color)
