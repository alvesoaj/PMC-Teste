require Dir.getwd+"/NeuralNetwork.rb"

number_of_entries = 4
layers = [15, 3]
learning_rate = 0.1
precision = 10 ** -5
momentum = 0.9

nn = NeuralNetwork::PMC.new(number_of_entries, layers, learning_rate, precision, momentum)

nn.set_training_and_output_samples('archives/training_samples.csv')

nn.train()

nn.production('archives/tabela.csv', false)