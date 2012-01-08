require Dir.getwd+"/NeuralNetwork.rb"

number_of_entries = 1
layers = [10, 1]
learning_rate = 0.01
precision = 10 ** -9
momentum = 0.5

nn = NeuralNetwork::PMC.new(number_of_entries, layers, learning_rate, precision, momentum)

nn.set_training_and_output_samples('archives/training_samples_est_sup.csv')

nn.train()

table = [[-1,5.5]]

nn.production(table, false)