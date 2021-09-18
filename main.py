from neural_network import NeuralNetwork


def main():

    NN = NeuralNetwork(2, 16, 1)
    print(len(NN.get_chr_format()))

    """
    n_bits = 65
    n_spec = 20
    n_gen = 100
    r_mut = 1 / n_bits
    r_cross = 0.85
    
    a = genetic_algorithm(objective_function, n_bits, n_gen, n_spec, r_cross, r_mut)

    print(a)
    input()
    """

    y = [-5.446671384297952, 9.224761607626114, 1.1380707763864333, -4.895358669092498, 2.5548091102198427,
         -6.224054108256166, 9.475750006650102, 7.779047820147891, 2.2736719637664073, -9.793903168216403,
         -9.149491264309109, -7.297385344204073, -5.045376036536315, 4.243761400446942, -9.604522722214437,
         -7.934594929330161, 6.427330253134048, 3.5219953667350303, -4.731489211592543, 5.787266302264673,
         -0.012634093884125619, -2.6074468269414393, -7.143797042219983, 3.8832789903640332, 4.923710129011125,
         -4.160836449075333, 7.675562772563971, -6.607199324415003, -2.405474410708834, 9.741574353392867,
         3.3221241142759226, -4.63485573416917, 4.316077243007008, 2.5834422061544906, -6.500789744962923,
         -6.120113198454433, -9.255956668574454, -7.2457440779930256, 6.098814594350898, -7.767045967595115,
         8.875991236924744, -1.5472342065238358, 3.6924985418396865, -9.520121791538365, -3.4320727709687526,
         7.282826536492127, -1.394626563561781, 3.7361829191405107, -5.467366021981772, 5.4757808351468675,
         -4.058778218446488, 5.19915909900573, 3.6261620769250733, -6.75448897699378, -4.160037805122581,
         3.8292938078311405, 0.8178715047516469, 8.57793855670835, 0.26539254813865476, 2.0248019490489497,
         0.34976507658539546, -4.024115235828409, 2.271328970138226, -1.2378372613642696, 0.24025781659554113]

    NN.fit_by_chr(y)
    print(NN.forward([0, 0]))
    print(NN.forward([0, 1]))
    print(NN.forward([1, 0]))
    print(NN.forward([1, 1]))


if __name__ == "__main__":
    main()