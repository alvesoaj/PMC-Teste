require 'csv'

@layers = [15, 3]
@number_of_entries = 3

@learning_rate = 0.1
@precision = 10 ** -6

@age = 0
 
CSV.open('archives/training_samples.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
 	@training_samples = []
 	@desired_output = []
   cvs.each do |row|
   	@training_samples << [row[0], row[1], row[2]]
   	@desired_output << row[3]
   end
end

@synaptic_weights = []

@layers.each_with_index do |layer, L|
	layer.times do |j|
		if L == 0
			#referente a entrada -1
			@synaptic_weights[L][j][0] = rand

			@number_of_entries.times do |i|
  				@synaptic_weights[L][j][i+1] = rand
  			end
  		else
  			#referente a entrada -1
			@synaptic_weights[L][j][0] = rand

			@layers[L-1].times do |i|
  				@synaptic_weights[L][j][i+1] = rand
  			end
  		end
  	end
end

begin
   puts("Inside the loop i = #$i" );
   $i +=1;
end until () < @precision