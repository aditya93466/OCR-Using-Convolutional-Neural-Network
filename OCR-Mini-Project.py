# Importing the libraries
"""

import tensorflow
tensorflow.__version__

import numpy as np
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

"""# Loading the datasets

## MNIST 0-9
"""

from tensorflow.keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data.shape, test_data.shape

28 * 28

train_labels.shape, test_labels.shape

train_data[0]

train_data[0].shape

train_labels[0]

train_labels

digits_data = np.vstack([train_data, test_data])
digits_labels = np.hstack([train_labels, test_labels])

digits_data.shape

digits_labels.shape

np.random.randint(0, digits_data.shape[0])

index = np.random.randint(0, digits_data.shape[0])
plt.imshow(digits_data[index], cmap='gray')
plt.title('Class: ' + str(digits_labels[index]));

sns.countplot(digits_labels);

"""## Kaggle A-Z"""

dataset_az = pd.read_csv('A_Z Handwritten Data.csv').astype('float32')
dataset_az

alphabet_data = dataset_az.drop('0', axis = 1)
alphabet_labels = dataset_az['0']

alphabet_data.shape, alphabet_labels.shape

alphabet_labels

alphabet_data = np.reshape(alphabet_data.values, (alphabet_data.shape[0], 28, 28))

alphabet_data.shape

index = np.random.randint(0, alphabet_data.shape[0])
plt.imshow(alphabet_data[index], cmap = 'gray')
plt.title('Class: ' + str(alphabet_labels[index]));

sns.countplot(alphabet_labels);

"""## Joining the datasets"""

digits_labels, np.unique(digits_labels)

alphabet_labels, np.unique(alphabet_labels)

alphabet_labels += 10

alphabet_labels, np.unique(alphabet_labels)

data = np.vstack([alphabet_data, digits_data])
labels = np.hstack([alphabet_labels, digits_labels])

data.shape, labels.shape

np.unique(labels)

data = np.array(data, dtype = 'float32')

data = np.expand_dims(data, axis = -1)

data.shape

"""# Pre-processing the data"""

data[0].min(), data[0].max()

data /= 255.0

data[0].min(), data[0].max()

np.unique(labels), len(np.unique(labels)) # softmax

le = LabelBinarizer()
labels = le.fit_transform(labels)

np.unique(labels)

labels

labels[0], len(labels[0])

labels[30000]

# OneHotEncoder
# A, B, C
# 0, 1, 2

# A, B, C
# 1, 0, 0
# 0, 1, 0
# 0, 0, 1

plt.imshow(data[0].reshape(28,28), cmap='gray')
plt.title(str(labels[0]));

classes_total = labels.sum(axis = 0)
classes_total

classes_total.max()

57825 / 6903

classes_weights = {}
for i in range(0, len(classes_total)):
  #print(i)
  classes_weights[i] = classes_total.max() / classes_total[i]

classes_weights

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 1, stratify = labels)

X_train.shape, X_test.shape

y_train.shape, y_test.shape

from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmentation = ImageDataGenerator(rotation_range = 10, zoom_range=0.05, width_shift_range=0.1,
                                  height_shift_range=0.1, horizontal_flip = False)

"""# Buiding the neural network


"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

network = Sequential()

network.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Conv2D(filters = 64, kernel_size=(3,3), activation='relu', padding='same'))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Conv2D(filters = 128, kernel_size=(3,3), activation='relu', padding='valid'))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Flatten())

network.add(Dense(64, activation = 'relu'))
network.add(Dense(128, activation = 'relu'))

network.add(Dense(36, activation='softmax'))

network.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

2 * 2 * 128

network.summary()

name_labels = '0123456789'
name_labels += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
name_labels = [l for l in name_labels]

print(name_labels)

"""# Training the neural network"""

file_model = 'custom_ocr.model'
epochs = 20
batch_size = 128

checkpointer = ModelCheckpoint(file_model, monitor = 'val_loss', verbose = 1, save_best_only=True)

len(X_train) // batch_size

history = network.fit(augmentation.flow(X_train, y_train, batch_size=batch_size),
                      validation_data = (X_test, y_test),
                      steps_per_epoch = len(X_train) // batch_size, epochs=epochs,
                      class_weight = classes_weights, verbose=1, callbacks=[checkpointer])

"""# Evaluating the neural network"""

X_test.shape

predictions = network.predict(X_test, batch_size=batch_size)

predictions

predictions[0]

len(predictions[0])

np.argmax(predictions[0])

name_labels[24]

y_test[0]

np.argmax(y_test[0])

name_labels[np.argmax(y_test[0])]

network.evaluate(X_test, y_test)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names = name_labels))

history.history.keys()

plt.plot(history.history['val_loss']);

plt.plot(history.history['val_accuracy']);

"""# Saving the neural network on Google Drive"""

network.save('network', save_format= 'h5')

from google.colab import drive
drive.mount('/content/drive')

!cp network /content/drive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Models/network

"""# Testing the neural network with images"""

from tensorflow.keras.models import load_model

loaded_network = load_model('/content/drive/MyDrive/Cursos - recursos/OCR with Python/Models/network')

loaded_network

loaded_network.summary()

import cv2
from google.colab.patches import cv2_imshow
img = cv2.imread('letter-m.jpg')
cv2_imshow(img)

img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape

cv2_imshow(gray)

value, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2_imshow(thresh)

value

thresh.shape

img = cv2.resize(thresh, (28,28))
cv2_imshow(img)

img.shape

img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis = -1)
img.shape

img = np.reshape(img, (1,28,28,1))
img.shape

prediction = loaded_network.predict(img)
prediction

np.argmax(prediction)

name_labels[22]
