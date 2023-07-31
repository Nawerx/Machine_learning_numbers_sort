import numpy as np


def load_dataset():
    """загружаем датасет с фотками для нейросети"""
    with np.load("mnist.npz") as f:
        # convert from RGB to Unit RGB
        x_train = f["x_train"].astype("float32") / 255

        # reshape from (60000, 28, 28) to (60000, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        print(x_train.shape)

        # labels
        y_train = f["y_train"]

        # convert to output layer format
        y_train = np.eye(10)[y_train]

        return x_train, y_train
