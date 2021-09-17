import numpy as np 
import random
import math

class dense_layer:
	def __init__(self, n_neurons, n_inputs):
		self._weight = [[random.uniform(-10, 10) for _ in range(n_inputs)] for _ in range(n_neurons)]
		self._bias = [random.uniform(-10, 10) for _ in range(n_neurons)]

	def activation_function(self, lista):
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
		assert len(self._weight) * len(self._weight[0]) + len(self._bias) == len(chromosome), 'Tamaño del cromosoma incorrecto'
		
		raw_weights = chromosome[:-len(self._bias)]
		self.set_bias(chromosome[-len(self._bias):])

		n_inputs = len(self._weight[0])
		n_neurons = len(self._weight)
		self._weight.clear()

		temp = 0
		for index in range(n_inputs, len(raw_weights) + n_inputs, n_inputs):
			self._weight += [raw_weights[temp:index]]
			temp = index

	def _ask_num(self):
		return len(self._weight) * len(self._weight[0]) + len(self._bias)

	def __str__(self):
		return f'(Neurons:{len(self._weight)}   Inputs:{len(self._weight[0])}   Outputs:{len(self._weight)})'






class Neural_network:
	def __init__(self, *h_layer):
		self._h_layers = h_layer
		self._neurons = list()
		for l in range(len(self._h_layers) - 1):
			layer = dense_layer(self._h_layers[l + 1], self._h_layers[l])
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
			num =  layer._ask_num()
			layer.fit_by_chr(chromosome[temp:temp + num])
			temp = temp + num


	def __str__(self):
		string = ''
		for i in self._neurons:
			string += str(i) + ' '
		return string


















def selection(population, scores, k=3):
	rand_sample_index = [random.randint(0, len(population) - 1) for _ in range(k)]
	best_spec = rand_sample_index[0]
	for spec in rand_sample_index:
		if scores[spec] < scores[best_spec]:
			best_spec = spec 
	return population[best_spec] 



def crossover(parent1, parent2, rate_cross=0.85):
	child1, child2 = parent1.copy(), parent2.copy()
	if random.random() < rate_cross:
		cut_index = random.randint(1, len(parent1) - 1)
		child1 = parent1[:cut_index] + parent2[cut_index:]
		child2 = parent2[:cut_index] + parent1[cut_index:]
	return child1, child2


def mutation(bitstring, rate_mutation):
	for i in range(len(bitstring)):
		if random.random() < rate_mutation:
			bitstring[i] =  random.uniform(-10, 10)
	return bitstring


def genetic_algorithm(objetive, n_bits, n_gen, n_spec, r_cross, r_mut):
	population = [[random.uniform(-10, 10) for _ in range(n_bits)] for _ in range(n_spec)]
	best, best_eval = population[0], objetive(population[0])

	for generation in range(n_gen):
		scores = [objetive(chromosome) for chromosome in population]

		print(f'Generation number:{generation} Best score:{best_eval}')
		for i in range(n_spec):
			if scores[i] < best_eval:
				best, best_eval = population[i], scores[i]
				print(f"<Best chromosome> ID:{population[i]} SCORE:{scores[i]} GENERATION:{generation}")

		parents = [selection(population, scores) for _ in range(n_spec)]
		children = list()
		for i in range(0, n_spec, 2):
			parent1, parent2 = parents[i], parents[i+1]
			child1, child2 = crossover(parent1, parent2, r_cross)
			children.append(mutation(child1, r_mut))
			children.append(mutation(child2, r_mut))
		population = children

	return best, best_eval


def funcion_objetivo(bitstring):
	# El batch son los casos de prueba y result es lo que debería devolver el perceptrón
	batch = [[0, 0],
			 [0, 1],
			 [1, 0],
			 [1, 1]]

	result = [0, 1, 1, 0]

	NN.fit_by_chr(bitstring)
 	
	score = [1 if NN.forward(batch[i])[0] == result[i] else 0 for i in range(len(batch))]
	return 1 - (sum(score)/4)

n_bits = 65
n_spec = 20
n_gen = 100
r_mut = 1/n_bits
r_cross = 0.85


NN = Neural_network(2, 16, 1)
print(len(NN.get_chr_format()))

'''
a = genetic_algorithm(funcion_objetivo, n_bits, n_gen, n_spec, r_cross, r_mut)

print(a)
input()
'''

y = [-5.446671384297952, 9.224761607626114, 1.1380707763864333, -4.895358669092498, 2.5548091102198427, -6.224054108256166, 9.475750006650102, 7.779047820147891, 2.2736719637664073, -9.793903168216403, -9.149491264309109, -7.297385344204073, -5.045376036536315, 4.243761400446942, -9.604522722214437, -7.934594929330161, 6.427330253134048, 3.5219953667350303, -4.731489211592543, 5.787266302264673, -0.012634093884125619, -2.6074468269414393, -7.143797042219983, 3.8832789903640332, 4.923710129011125, -4.160836449075333, 7.675562772563971, -6.607199324415003, -2.405474410708834, 9.741574353392867, 3.3221241142759226, -4.63485573416917, 4.316077243007008, 2.5834422061544906, -6.500789744962923, -6.120113198454433, -9.255956668574454, -7.2457440779930256, 6.098814594350898, -7.767045967595115, 8.875991236924744, -1.5472342065238358, 3.6924985418396865, -9.520121791538365, -3.4320727709687526, 7.282826536492127, -1.394626563561781, 3.7361829191405107, -5.467366021981772, 5.4757808351468675, -4.058778218446488, 5.19915909900573, 3.6261620769250733, -6.75448897699378, -4.160037805122581, 3.8292938078311405, 0.8178715047516469, 8.57793855670835, 0.26539254813865476, 2.0248019490489497, 0.34976507658539546, -4.024115235828409, 2.271328970138226, -1.2378372613642696, 0.24025781659554113]

NN.fit_by_chr(y)
print(NN.forward([0,0]))
print(NN.forward([0,1]))
print(NN.forward([1,0]))
print(NN.forward([1,1]))


