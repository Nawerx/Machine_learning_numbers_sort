
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import utils
matplotlib.use('Qt5Agg')
# class Point:
#     min_value = 0
#     max_value = 1000
#
#     @classmethod
#     def validate(cls, arg):
#         return cls.min_value <= arg <= cls.max_value
#
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#
#
#
#
# pt = Point(3,5)
# print(Point.validate(5))

images, labels = utils.load_dataset()

'''генерируем рандомные веса для нейронов'''

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

'''генерируем вектор биас  для нейронов'''

bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

# количество епох и ошибок нейросети
epochs = 5
e_loss = 0
e_correct = 0
learning_rate = 0.01
for epoch in range(epochs):
    print(f"Epoch №{epoch}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # Forward propagation (to hidden layer)
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1/(1 + np.exp(-hidden_raw)) # sigmoid activation function

        # Forward propagation (to output layer)
        output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
        output = 1 / (1 + np.exp(-output_raw))

        # Loss / error calculation
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # Backpropagation (output layer)
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # Backpropagation (hidden layer)
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

        # Finish Backpropagation

    print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
    e_loss = 0
    e_correct = 0


import random

test_image = random.choice(images)

image = np.reshape(test_image, (-1, 1))

# Forward propagation (to hidden layer)
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1/(1 + np.exp(-hidden_raw)) # sigmoid activation function

# Forward propagation (to output layer)
output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"NN suggested number is {output.argmax()}")
plt.show()


