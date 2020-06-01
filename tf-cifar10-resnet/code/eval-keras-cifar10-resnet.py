import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('cifar10-resnet.h5')
# model.summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x = x_train
# y = y_train
x = x_test
y = y_test

# Pre-processing
m = np.mean(x, axis=(1, 2, 3))
s = np.std(x, axis=(1, 2, 3))
x = (x - m[:, np.newaxis, np.newaxis, np.newaxis])/s[:, np.newaxis, np.newaxis, np.newaxis]

# Setting
batch_size = 100

iterations = int(len(x)/batch_size)
result = np.zeros(len(x))

for i in range(0, iterations):
    img = x[batch_size * i:batch_size * (i + 1)]
    result[batch_size * i:batch_size * (i + 1)] = np.argmax(model.predict(img), axis=1)

num_correct = np.sum(result.flatten() == y.flatten())
print('Accuracy: %f [%%] (%d / %d)' % (num_correct/len(x)*100., num_correct, len(x)))
