require 'csv'

@training_samples = []
@desired_output = []

@layers = [15, 3]
@number_of_entries = 3

@learning_rate = 0.1
@precision = 10 ** -6

@age = 0
 
CSV.open('archives/training_samples.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
  cvs.each do |row|
    @training_samples << [-1, row[0], row[1], row[2]]
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
  @old_error = @error
  @training_samples.each_with_index do |ts, si|
    # Preenchendo o cubo I
    @layers.each_with_index do |layer, L|
      layer.times do |j|
        if L == 0
          @number_of_entries.times do |i|
            @I[L][j][i] = 0
            ts.size.times do |d|
              @I[L][j][i] += @synaptic_weights[L][j][i] * ts[d]
            end
            # Preenchendo o cubo Y
            @Y[L][j][i+1] = Math.tanh(@I[L][j][i])
          end
          @Y[L][j][0] = -1
        else
          @number_of_entries.times do |i|
            @I[L][j][i] = 0
            ts.size.times do |d|
              @I[L][j][i] += @synaptic_weights[L][j][i] * ts[d]
            end
            # Preenchendo o cubo Y
            if L != (@layers.size -1)
              @Y[L][j][i+1] = Math.tanh(@I[L][j][i])
            else
              @Y[L][j][i] = Math.tanh(@I[L][j][i])
            end
          end
          if L != (@layers.size -1)
            @Y[L][j][0] = -1
          end
        end
      end
    end

    #gradiente
    @gradient = []
  end
end until (@error - @old_error) < @precision