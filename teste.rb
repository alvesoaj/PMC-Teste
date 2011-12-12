require Dir.getwd+"/NeuralNetwork.rb"

nn = NeuralNetwork::PMC.new

nn.set_training_and_output_samples()

nn.train()

nn.production(nil, false)