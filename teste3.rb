require Dir.getwd+"/NeuralNetwork.rb"

number_of_entries = 3
layers = [10, 1]
learning_rate = 0.1
precision = 10 ** -6
momentum = 0.0

nn = NeuralNetwork::PMC.new(number_of_entries, layers, learning_rate, precision, momentum)

nn.set_training_and_output_samples('archives/training_samples_daniel.csv')

nn.train()

table = [[-1,0.0611,0.2860,0.7464]]

nn.production(table, false)