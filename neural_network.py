from dense_layer import DenseLayer


class NeuralNetwork:
    def __init__(self, *h_layer):
        self._h_layers = h_layer
        self._neurons = list()
        for L in range(len(self._h_layers) - 1):
            layer = DenseLayer(self._h_layers[L + 1], self._h_layers[L])
            self._neurons.append(layer)

    def forward(self, inputs):
        I_O = inputs
        for layer in self._neurons:
            I_O = layer.forward(I_O)

        return I_O

    def get_chr_format(self):
        return [j for i in self._neurons for j in i.get_chr_format()]

    def fit_by_chr(self, chromosome):
        temp = 0
        for layer in self._neurons:
            num = layer.ask_num()
            layer.fit_by_chr(chromosome[temp:temp + num])
            temp = temp + num

    def __str__(self):
        string = ''
        for i in self._neurons:
            string += str(i) + ' '
        return string
