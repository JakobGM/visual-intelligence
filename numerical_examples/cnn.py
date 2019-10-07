import itertools
from typing import Any, Callable, List, Tuple

from matplotlib import pyplot as plt

import numpy as np


def plot_matrix(matrix: np.ndarray, ax=None) -> None:
    if not ax:
        _, ax = plt.subplots(1, 1)

    ax.imshow(matrix)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    for row, col in itertools.product(
        range(matrix.shape[0]),
        range(matrix.shape[1]),
    ):
        ax.text(
            x=col,
            y=row,
            s=matrix[row, col],
            ha="center",
            va="center",
            color="w",
            size=30,
        )


def windows(
    array: np.ndarray,
    window_shape: Tuple[int, int],
    stride: Tuple[int, int],
    apply: Callable[[np.ndarray], Any],
) -> List[List[np.ndarray]]:
    array_height, array_width = array.shape
    stride_height, stride_width = stride
    window_height, window_width = window_shape

    rows = []
    for row in range(0, array_height, stride_height):
        if row + window_height > array_height:
            continue

        cols = []
        for col in range(0, array_width, stride_width):
            if col + window_width > array_width:
                continue

            window = array[
                row:row + window_height,
                col:col + window_width,
            ]
            window = apply(window) if apply else window

            cols.append(window)

        if cols:
            rows.append(cols)

    return np.array(rows)


class ActivationFunction:

    @staticmethod
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)


class Kernel:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.array(weights)
        self.shape = self.weights.shape

    def __mul__(self, receptive_field: np.ndarray) -> float:
        assert receptive_field.shape == self.shape
        elementwise_multiplication = self.weights * receptive_field
        summarization = elementwise_multiplication.sum()
        return summarization

    __rmul__ = __mul__

    def convolute(
        self,
        array: np.ndarray,
        stride: Tuple[int, int],
        padding: int = 0,
        activation_function: Callable[
            [np.ndarray],
            np.ndarray
        ] = ActivationFunction.identity,
        bias: float = 0.0,
    ) -> np.ndarray:
        if padding:
            array = np.pad(array, pad_width=padding)

        return windows(
            array=array,
            window_shape=self.shape,
            stride=stride,
            apply=lambda x: activation_function(self * x + bias),
        )

    def plot(self) -> None:
        plot_matrix(self.weights)

    def plot_convolution(self, receptive_field: np.ndarray) -> None:
        figure, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 8))
        elementwise_multiplication = self.weights * receptive_field
        summarization = elementwise_multiplication.sum()

        ax1.set_title("Kernel")
        plot_matrix(self.weights, ax=ax1)

        ax2.set_title("Receptive Field")
        plot_matrix(receptive_field, ax=ax2)

        ax3.set_title("Multiplication")
        plot_matrix(elementwise_multiplication, ax=ax3)

        ax4.set_title("Sum")
        plot_matrix(np.array([[summarization]]), ax=ax4)


class MaxPool:
    def __init__(
        self,
        shape: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> None:
        self.shape = shape
        self.stride = stride

    def pool(self, array):
        return windows(
            array=array,
            window_shape=self.shape,
            stride=self.stride,
            apply=lambda x: x.max()
        )
