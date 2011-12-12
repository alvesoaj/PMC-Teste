# encoding: utf-8
module NeuralNetwork
	
	require 'csv'

	FUNC_ADJ = 0.5

	# Função tangente hiperbôlica
	def self.hyperbolic_tangent(value)
		return Math.tanh(value)
	end

	# Derivada da função Tangente Hiperbôlica
	def self.hyperbolic_tangent_derived(value)
	  return 1.0 - value ** 2
	end

	# g(u) = 1/(1+e^(-bu)) Função Logistica
	def self.logistic_function(value)
	    return 1 / (1 + (Math::E ** (-1 * FUNC_ADJ * value)))
	end

	# Derivada da função logistica
	def self.logistic_function_derived(value)
	    return FUNC_ADJ * (logistic_function(value)) * (1 - logistic_function(value))
	end

	class PMC
	  	attr_accessor :number_of_entries, :layers, :learning_rate, :precision, :momentum

	 
	  	def initialize(number_of_entries = 2, layers = [2,1], learning_rate = 0.1, precision = 10 ** -5, momentum = 0.0)
	   	@number_of_entries = number_of_entries
	   	@layers = layers
	   	@learning_rate = learning_rate
			@precision = precision
			@momentum = momentum
			# Iniciando os pesos sinapticos
			init_synaptic_weights()
	  	end

	  	#inicializador de pesos sinápticos
	  	def init_synaptic_weights()
	  		@synaptic_weights = Array.new
			@null_synaptic_weights = Array.new

	  		@layers.each_with_index do |layer, _L|
			  	@synaptic_weights[_L] = Array.new
				@null_synaptic_weights[_L] = Array.new

				if _L == 0
				   layer.times do |j|
				      @synaptic_weights[_L][j] = Array.new
						@null_synaptic_weights[_L][j] = Array.new

				      (@number_of_entries + 1).times do |i|
				      	@synaptic_weights[_L][j][i] = rand
				      	@null_synaptic_weights[_L][j][i] = 0
				      end
				   end
				else
				   layer.times do |j|
				      @synaptic_weights[_L][j] = Array.new
						@null_synaptic_weights[_L][j] = Array.new

				      (@layers[_L-1] + 1).times do |i|
							@synaptic_weights[_L][j][i] = rand
				      	@null_synaptic_weights[_L][j][i] = 0
						end
				   end
			  	end
			end

			@ws = Array.new
			@ws[0] = @null_synaptic_weights.clone
			@ws[1] = @null_synaptic_weights.clone
	  	end

	  	def get_synaptic_weights()
	  		csv = CSV.open('archives/synaptic_weights.csv', 'r', {:col_sep => ',', :converters => :float}).to_a

	  		@synaptic_weights = Array.new

			@layers.each_with_index do |layer, _L|
				neuro = []
				soma = 0
				(0..(_L-1)).each do |t|
					soma += @layers[t]
				end

				layer.times do |j|
					entry = []
					entry = csv[soma+j]
					neuro << entry
				end
				@synaptic_weights << neuro
			end
	  	end

	  	def set_training_and_output_samples(path = nil)
	  		if path.nil?
	  			training_samples = [[-1,0,1],[-1,0,0],[-1,1,0],[-1,1,1]]
	  			desired_output = [[1],[0],[1],[0]]
	  		else
		  		training_samples = Array.new
				desired_output = Array.new
				CSV.open(Dir.getwd+"/"+path, 'r', {:col_sep => ',', :converters => :float}) do |cvs|
					cvs.each do |row|
						line = Array.new
						line << -1
						@number_of_entries.times do |i|
							line << row[i]
						end
				   	training_samples << line

				   	line = Array.new
				   	@layers.last.times do |i|
				   		line << row[@number_of_entries+i]
				   	end
				   	desired_output << line
				 	end
				end
			end
	  		@training_samples = training_samples
	  		@desired_output = desired_output
	  	end

	  	def train()
	  		age = 0
	  		error = 0
	  		old_error = 0

	  		time = Time.now

	  		error_arch = CSV.open("archives/errors.csv", "wb")
	  		puts "Iniciando treinamento"

	  		begin
	  			if (age % 100) == 0 && age > 0
				  	puts "Entrando na era: "+age.to_s
				  	puts (error - old_error).abs
				end
			  	old_error = error

			  	@training_samples.each_with_index do |ts, ti|
			  		@I = Array.new
	  				@Y = Array.new
	  				@gradient = Array.new

			    	forward(ts)

			    	backward(ts, ti)
			  	end

			  	age += 1

			  	error = get_error()
			  	error_arch << [(error - old_error).abs]
			end until ((error - old_error).abs <= @precision) || (age > 100000)

			puts "Treinamento finalizado em "+(Time.now - time).to_i.to_s+" segundos na era "+age.to_s+". Pesos salvos com sucesso!"

			CSV.open("archives/synaptic_weights.csv", "wb") do |csv|
				@layers.each_with_index do |layer, _L|
			   	layer.times do |j|
			      	csv << @synaptic_weights[_L][j]
			    	end
			 	end
			end
	  	end

	  	def production(value = nil, load_weights = true)
	  		if load_weights
	  			puts "Carregando pessos sinàpticos do arquivo."
	  			get_synaptic_weights()
	  		end

	  		if value.nil?
	  			entries = [[-1,0,1],[-1,0,0],[-1,1,0],[-1,1,1]]
	  		elsif (value.is_a? Array)
	  			entries = value
	  		else
	  			entries = Array.new
				CSV.open(Dir.getwd+"/"+value, 'r', {:col_sep => ',', :converters => :float}) do |cvs|
				  	cvs.each do |row|
				  		line = Array.new
						line << -1
						@number_of_entries.times do |i|
							line << row[i]
						end
				   	entries << line
				  	end
				end
	  		end

	  		puts "Produção:"
	  		entries.each do |inputs|
			  	@I = Array.new
	  			@Y = Array.new

	  			forward(inputs)

			  	result = Array.new
			  	@layers[@layers.size-1].times do |j|
			   	if @Y[@layers.size-1][j] >= 0.5
			   		result << 1
			    	else
			      	result << 0
			    	end
			  	end

			  	puts @Y[@layers.size-1].join(" | ")
			  	puts result.join(" | ")
			end
	  	end

	  	def forward(training_sample)
	  		# Preenchendo as matrizes I e Y
		   @layers.each_with_index do |layer, _L|
		   	@I[_L] = Array.new
		   	@Y[_L] = Array.new

		      if _L == 0 #-------------------------------------------------- _L primeiro
					layer.times do |j|
						soma = 0
						(@number_of_entries + 1).times do |i|
							soma += @synaptic_weights[_L][j][i] * training_sample[i]
						end
						@I[_L][j] = soma
					end

		        	@Y[_L][0] = -1
		        	layer.times do |j|
		         	@Y[_L][j+1] = NeuralNetwork::logistic_function(@I[_L][j])
		        	end
		      elsif _L == (@layers.size - 1) #--------------------------------- _L último
		        	layer.times do |j|
		         	soma = 0
		          	(@layers[_L-1] + 1).times do |i|
		            	soma += @synaptic_weights[_L][j][i] * @Y[_L-1][i]
		          	end
		          	@I[_L][j] = soma
		        	end

		        	layer.times do |j|
		         	@Y[_L][j] = NeuralNetwork::logistic_function(@I[_L][j])
		        	end
		      else #--------------------------------------------------------- _L intermediários
		        	layer.times do |j|
		         	soma = 0
		          	(@layers[_L-1] + 1).times do |i|
		            	soma += @synaptic_weights[_L][j][i] * @Y[_L-1][i]
		          	end
		          	@I[_L][j] = soma
		        end

		        	@Y[_L][0] = -1
		        	layer.times do |j|
		          	@Y[_L][j+1] = NeuralNetwork::logistic_function(@I[_L][j])
		        	end
		      end
		    end
	  	end

	  	def backward(training_sample, training_index)
		   (@layers.size-1).downto 0 do |_L|
		   	@gradient[_L] = Array.new

		      if _L == (@layers.size - 1) #--------------------------------- _L último
		        	#gradiente
		        	@layers[_L].times do |j|
		          	@gradient[_L][j] = (@desired_output[training_index][j] - @Y[_L][j]) * NeuralNetwork::logistic_function_derived(@I[_L][j])
		        	end		    

		        	#ajustando pesos sinápticos
		        	@layers[_L].times do |j|
		         	(@layers[_L-1] + 1).times do |i|
		            	@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + (@momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f)) + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
		          	end
		        	end
		      elsif _L == 0 #-------------------------------------------------- _L primeiro
		        	#gradiente
		        	neuro = []
		        	@layers[_L].times do |j|
		         	soma = 0
		          	@layers[_L+1].times do |k|
		            	soma += @gradient[_L+1][k] * @synaptic_weights[_L+1][k][j]
		          	end
		          	@gradient[_L][j] = soma * NeuralNetwork::logistic_function_derived(@I[_L][j])
		        	end

		        	#ajustando pesos sinápticos
		        	@layers[_L].times do |j|
		         	(@number_of_entries + 1).times do |i|
		            	@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + (@momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f)) + @learning_rate * @gradient[_L][j] * training_sample[i]
		          	end
		        	end
		      else #--------------------------------------------------------- _L intermediários
		        	#gradiente
		        	neuro = []
		        	@layers[_L].times do |j|
		         	soma = 0
		          	@layers[_L+1].times do |k|
		            	soma += @gradient[_L+1][k] * @synaptic_weights[_L+1][k][j]
		          	end
		          	@gradient[_L][j] = soma * NeuralNetwork::logistic_function_derived(@I[_L][j])
		        	end

		        	#ajustando pesos sinápticos
		        	@layers[_L].times do |j|
		         	(@layers[_L-1] + 1).times do |i|
		            	@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + (@momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f)) + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
		          	end
		        	end
		      end
		   end

		   @ws[1] = @ws[0].clone
		   @ws[0] = @synaptic_weights.clone
	  	end

	  	def get_error()
	  		soma_tot = 0
	  		@I = Array.new
	  		@Y = Array.new
			@training_samples.each_with_index do |ts, ti|
				forward(ts)

			   soma = 0
			   @layers.last.times do |j|
			   	soma += (@desired_output[ti][j] - @Y[@layers.size-1][j]) ** 2
			   end
			   soma_tot += soma / 2
			end
			
			return soma_tot / @training_samples.size
	  	end
	end
end