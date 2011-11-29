require 'csv'
require Dir.getwd+"/NeuralNetwork.rb"

number_of_entries = 3
layers = [10, 1]
learning_rate = 0.1
precision = 10 ** -6
momentum = 0.0

nn = NeuralNetwork::PMC.new(number_of_entries, layers, learning_rate, precision, momentum)

training_samples = Array.new
desired_output = Array.new
CSV.open('archives/training_samples_daniel.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
	cvs.each do |row|
   	training_samples << [-1, row[0], row[1], row[2]]
   	desired_output << [row[3]]
 	end
end

nn.set_training_and_otput_samples(training_samples, desired_output)

nn.train()

table = [[-1,0.0611,0.2860,0.7464]]

nn.production(table)