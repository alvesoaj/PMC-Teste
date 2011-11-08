require 'csv'
 
CSV.open('archives/training_samples.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
 	@training_samples = []
 	@desired_output = []
   cvs.each do |row|
   	@training_samples << [row[0], row[1], row[2]]
   	@desired_output << row[3]
   end
end