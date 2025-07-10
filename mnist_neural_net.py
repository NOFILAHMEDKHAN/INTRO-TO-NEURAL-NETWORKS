import tensorflow as tf

from tensorflow.keras import layers, models

import matplotlib.pyplot as plt

# Load the MNIST dataset

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to a [0, 1] range

x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model

model = models.Sequential([

layers.Flatten(input_shape=(28, 28)),

layers.Dense(128, activation='relu'),

layers.Dropout(0.2),

layers.Dense(10, activation='softmax')])

# Compile the model

model.compile(optimizer='adam',

loss='sparse_categorical_crossentropy',

metrics=['accuracy'])

# Train the model

history = model.fit(x_train, y_train, epochs=5,
validation_data=(x_test, y_test))

# Evaluate the model on the test data

test_loss, test_accuracy = model.evaluate(x_test, y_test,
verbose=2)

print(f'\nTest accuracy: {test_accuracy:.4f}')

# Plot accuracy and loss

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Make predictions on the test data

predictions = model.predict(x_test)

print(f'Predicted label for the first test sample:
{predictions[0].argmax()}')

print(f'Actual label for the first test sample: {y_test[0]}')

