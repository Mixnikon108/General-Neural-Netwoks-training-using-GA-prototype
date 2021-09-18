import random
import numpy as np


class DenseLayer:

    def __init__(self, n_neurons, n_inputs):
        self._weight = [[random.uniform(-10, 10) for _ in range(n_inputs)] for _ in range(n_neurons)]
        self._bias = [random.uniform(-10, 10) for _ in range(n_neurons)]

    @staticmethod
    def activation_function(lista):
        # Discrete activation
        return [1 if x > 0 else 0 for x in lista]

    def forward(self, inputs):
        return self.activation_function(np.dot(self._weight, inputs) + self._bias)

    def get_weights(self):
        return self._weight

    def get_bias(self):
        return self._bias

    def set_weights(self, arg):
        self._weight = arg

    def set_bias(self, arg):
        self._bias = arg

    def get_chr_format(self):
        return [j for i in self._weight for j in i] + self._bias

    def fit_by_chr(self, chromosome):
        # Comprobar que tiene la longitud adecuada
        assert len(self._weight) * len(self._weight[0]) + len(self._bias) == len(
            chromosome), 'Tamaño del cromosoma incorrecto'

        raw_weights = chromosome[:-len(self._bias)]
        self.set_bias(chromosome[-len(self._bias):])

        n_inputs = len(self._weight[0])
        self._weight.clear()

        temp = 0
        for index in range(n_inputs, len(raw_weights) + n_inputs, n_inputs):
            self._weight += [raw_weights[temp:index]]
            temp = index

    def ask_num(self):
        return len(self._weight) * len(self._weight[0]) + len(self._bias)

    def __str__(self):
        return f'(Neurons:{len(self._weight)}   Inputs:{len(self._weight[0])}   Outputs:{len(self._weight)})'
