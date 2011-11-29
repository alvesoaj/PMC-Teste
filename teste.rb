require Dir.getwd+"/NeuralNetwork.rb"

nn = NeuralNetwork::PMC.new

nn.set_training_and_otput_samples()

nn.train()

nn.production()