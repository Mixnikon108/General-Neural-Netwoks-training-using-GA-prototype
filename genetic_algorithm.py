import random

from neural_network import NeuralNetwork


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
            bitstring[i] = random.uniform(-10, 10)
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
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2, r_cross)
            children.append(mutation(child1, r_mut))
            children.append(mutation(child2, r_mut))
        population = children

    return best, best_eval


def objective_function(bitstring, nn: NeuralNetwork):
    # El batch son los casos de prueba y result es lo que debería devolver el perceptrón
    batch = [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]

    result = [0, 1, 1, 0]

    nn.fit_by_chr(bitstring)

    score = [1 if nn.forward(batch[i])[0] == result[i] else 0 for i in range(len(batch))]
    return 1 - (sum(score) / 4)
