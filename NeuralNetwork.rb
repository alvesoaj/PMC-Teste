# encoding: utf-8
module NeuralNetwork
	FUNC_ADJ = 0.5

	# Função tangente hiperbôlica
	def hyperbolic_tangent(value)
		return Math.tanh(value)
	end

	# Derivada da função Tangente Hiperbôlica
	def derived_hyperbolic_tangent(value)
	  return 1.0 - value ** 2
	end

	# g(u) = 1/(1+e^(-bu)) Função Logistica
	def logistic_function(value)
	    return 1 / (1 + (Math::E ** (-1 * FUNC_ADJ * value)))
	end

	# Derivada da função logistica
	def derived_logistic_function(value)
	    return FUNC_ADJ * (logistic_function(value)) * (1 - logistic_function(val_fun))
	end

	class PMC
	  	attr_accessor :number_of_entries, :layers, :learning_rate, :precision, :momentum

	 
	  	def initialize(number_of_entries = 2, layers = [2,1], learning_rate = 0.1, precision = 10 ** -8, momentum = 0.0)
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
			@ws[0] = @null_synaptic_weights[_L][j][i].clone
			@ws[1] = @null_synaptic_weights[_L][j][i].clone
	  	end

	  	def set_training_samples(ts = [[0,1],[0,0],[1,0],[1,1]], des_o = [[1],[0],[1],[0]])
	  		@training_samples = ts
	  		@desired_output = des_o
	  	end

	  	def train()
	  		age = 0
	  		error = 0
	  		old_error = 0
	  		@I = Array.new
	  		@Y = Array.new
	  		@gradient = Array.new

	  		begin
			  	puts "Entrando na era: "+age.to_s
			  	old_error = error

			  	@training_samples.each_with_index do |ts, ti|
			    	forward(ts)

			    	backward(ti)
			  
			  # -------------------------------------------------------------------------------------------- Calculando erro
			  
			  #puts @I.map{|x| x.join(" | ")}
			  #puts @Y.map{|x| x.join(" | ")}

			  soma_tot = 0
			  @training_samples.size.times do |k|
			    @all_I = []
			    @all_Y = []
			    #------------------------------------------------------------------------------------------------ forward
			    # Preenchendo as matrizes I e Y
			    @layers.each_with_index do |layer, _L|
			      if _L == 0 #-------------------------------------------------- _L primeiro
			        #matriz I
			        neuro = []
			        layer.times do |j|
			          soma = 0
			          (@number_of_entries + 1).times do |i|
			            soma += @synaptic_weights[_L][j][i] * @training_samples[k][i]
			            #puts soma.to_s+" | "+@synaptic_weights[_L][j][i].to_s+" | "+ts[i].to_s
			          end
			          neuro << soma
			        end

			        @all_I << neuro

			        #matriz Y
			        neuro = []
			        neuro << -1
			        layer.times do |j|
			          #neuro << Math.tanh(@I[_L][j])
			          neuro << funcao_sigmoid(@I[_L][j])
			          #puts funcao_sigmoid(@I[_L][j])  
			        end

			        @all_Y << neuro
			      elsif _L == (@layers.size - 1) #--------------------------------- _L último
			        #matriz I
			        neuro = []
			        layer.times do |j|
			          soma = 0
			          (@layers[_L-1] + 1).times do |i|
			            soma += @synaptic_weights[_L][j][i] * @Y[_L-1][i]
			            #puts soma.to_s+" | "+@synaptic_weights[_L][j][i].to_s+" | "+@Y[_L-1][i].to_s
			          end
			          neuro << soma
			        end

			        @all_I << neuro

			        #matriz Y
			        neuro = []
			        layer.times do |j|
			          #neuro << Math.tanh(@I[_L][j])
			          neuro << funcao_sigmoid(@I[_L][j])
			          #puts funcao_sigmoid(@I[_L][j])
			        end

			        @all_Y << neuro
			      else #--------------------------------------------------------- _L intermediários
			        #matriz I
			        neuro = []
			        layer.times do |j|
			          soma = 0
			          (@layers[_L-1] + 1).times do |i|
			            soma += @synaptic_weights[_L][j][i] * @Y[_L-1][i]
			          end
			          neuro << soma
			        end

			        @all_I << neuro

			        #matriz Y
			        neuro = []
			        neuro << -1
			        layer.times do |j|
			          #neuro << Math.tanh(@I[_L][j])
			          neuro << funcao_sigmoid(@I[_L][j])
			        end

			        @all_Y << neuro
			      end
			    end
			    #------------------------------------------------------------------ Calcular erro local 
			    soma = 0
			    #puts @I[@layers.size-1].join(" , ")
			    #puts @Y[@layers.size-1].join(" , ")
			    @layers.last.times do |j|
			      #puts @desired_output[ti][j].to_s+" - "+@Y[@layers.size-1][j].to_s
			      soma += (@desired_output[k][j] - @all_Y[@layers.size-1][j]) ** 2
			    end
			    soma_tot += soma / 2
			  end
			  @error = soma_tot / @training_samples.size
			  
			  # -------------------------------------------------------------------------------------------- Contando eras
			  @age += 1
			  @error_arch << [(@error - @old_error).abs]
			  #puts @error.to_s+" | "+@old_error.to_s
			  #puts @errors.join(" - ")
			  #puts @I.join(" - ")
			end until ((@error - @old_error).abs <= @precision) || (@age > 100000)
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
		         	@Y[_L][j+1] = logistic_function(@I[_L][j])
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
		         	@Y[_L][j] = logistic_function(@I[_L][j])
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
		          	@Y[_L][j+1] = logistic_function(@I[_L][j])
		        	end
		      end
		    end
	  	end

	  	def backward(training_index)
		   (@layers.size-1).downto 0 do |_L|
		   	@gradient[_L] = Array.new

		      if _L == (@layers.size - 1) #--------------------------------- _L último
		        	#gradiente
		        	@layers[_L].times do |j|
		          	@gradient[_L][j] = (@desired_output[training_index][j] - @Y[_L][j]) * derived_logistic_function(@I[_L][j])
		        	end		    

		        	#ajustando pesos sinápticos
		        	@layers[_L].times do |j|
		         	(@layers[_L-1] + 1).times do |i|
		            	@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f) + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
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
		          	@gradient[_L][j] = soma * derived_logistic_function(@I[_L][j])
		        	end

		        	#ajustando pesos sinápticos
		        	@layers[_L].times do |j|
		         	(@number_of_entries + 1).times do |i|
		            	@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f) + @learning_rate * @gradient[_L][j] * ts[i]
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
		          	@gradient[_L][j] = soma * derived_logistic_function(@I[_L][j])
		        	end

		        	#ajustando pesos sinápticos
		        	@layers[_L].times do |j|
		         	(@layers[_L-1] + 1).times do |i|
		            	@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f) + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
		          	end
		        	end
		      end
		    end

		    @ws[1] = @ws[0].clone
		    @ws[0] = @synaptic_weights.clone
		  end
	  	end
	end
end