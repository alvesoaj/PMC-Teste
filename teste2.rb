require 'csv'
require Dir.getwd+"/NeuralNetwork.rb"

number_of_entries = 4
layers = [15, 3]
learning_rate = 0.1
precision = 10 ** -8
momentum = 0.9

nn = NeuralNetwork::PMC.new(number_of_entries, layers, learning_rate, precision, momentum)

training_samples = Array.new
desired_output = Array.new
CSV.open('archives/training_samples.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
	cvs.each do |row|
   	training_samples << [-1, row[0], row[1], row[2], row[3]]
   	desired_output << [row[4], row[5], row[6]]
 	end
end

nn.set_training_and_otput_samples(training_samples, desired_output)

nn.train()

table = Array.new
CSV.open('archives/tabela.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
  	cvs.each do |row|
   	table << [-1, row[0], row[1], row[2], row[3]]
  	end
end

nn.production(table)