require Dir.getwd+"/NeuralNetwork.rb"

nn = NeuralNetwork::PMC.new

nn.set_training_samples()

nn.train()