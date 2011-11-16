require 'csv'

@training_samples = []
@desired_output = []

@layers = [15, 3]
@number_of_entries = 4

@learning_rate = 0.1
@precision = 10 ** -6

@age = 0

@error = 0.0

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dtanh(x)
  return 1.0 - x**2
end
 
CSV.open('archives/training_samples.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
  cvs.each do |row|
    @training_samples << [-1, row[0], row[1], row[2], row[3]]
    @desired_output << [row[4], row[5], row[6]]
  end
end

@synaptic_weights = []

@layers.each_with_index do |layer, _L|
  #puts "layer: "+_L.to_s
  neuro = []
	if _L == 0
    layer.times do |j|
      #puts "neuronio: "+j.to_s
      entry = []
      (@number_of_entries + 1).times do |i|
        #puts "entrada: "+i.to_s
        entry << rand
      end
      neuro << entry
    end
	else
    layer.times do |j|
      #puts "neuronio: "+j.to_s
      entry = []
      (@layers[_L-1] + 1).times do |i|
        #puts "entrada: "+i.to_s
				entry << rand
			end
      neuro << entry
    end
  end
  @synaptic_weights << neuro
end

@I = []
@Y = []

begin
  puts "Entrando na era: "+@age.to_s
  @old_error = @error
  @errors = []

  @training_samples.each_with_index do |ts, ti|
    #------------------------------------------------------------------------------------------------ forward
    # Preenchendo as matrizes I e Y
    @layers.each_with_index do |layer, _L|
      if _L == 0 #-------------------------------------------------- _L primeiro
        #matriz I
        neuro = []
        layer.times do |j|
          soma = 0
          (@number_of_entries + 1).times do |i|
            soma += @synaptic_weights[_L][j][i] * ts[i]
          end
          neuro << soma
        end

        @I << neuro

        #matriz Y
        neuro = []
        neuro << -1
        (1..layer).each do |j|
          neuro << Math.tanh(@I[_L][j-1])
        end

        @Y << neuro
      elsif _L == (@layers.size - 1) #--------------------------------- _L último
        #matriz I
        neuro = []
        layer.times do |j|
          soma = 0
          @layers[_L-1].times do |i|
            soma += @synaptic_weights[_L][j][i] * @Y[_L-1][j]
          end
          neuro << soma
        end

        @I << neuro

        #matriz Y
        neuro = []
        layer.times do |j|
          neuro << Math.tanh(@I[_L][j])
        end

        @Y << neuro
      else #--------------------------------------------------------- _L intermediários
        #matriz I
        neuro = []
        layer.times do |j|
          soma = 0
          @layers[_L-1].times do |i|
            soma += @synaptic_weights[_L][j][i] * @Y[_L-1][j]
          end
          neuro << soma
        end

        @I << neuro

        #matriz Y
        neuro = []
        neuro << -1
        (1..layer).each do |j|
          neuro << Math.tanh(@I[_L][j-1])
        end

        @Y << neuro
      end
    end

    #------------------------------------------------------------------ Calcular erro local
    soma = 0
    @layers.last.times do |j|
      soma += (@desired_output[ti][j] - @Y[@layers.size-1][j]) ** 2
    end
    @errors[ti] = soma / 2

    #------------------------------------------------------------------------------------------------- backward
    @gradient = []

    (@layers.size-1).downto 0 do |_L|
      if _L == (@layers.size - 1) #--------------------------------- _L último
        #gradiente
        neuro = []
        @layers[_L].times do |j|
          neuro << (@desired_output[ti][j] - @Y[_L][j]) * dtanh(@I[_L][j])
        end

        @gradient[_L] = neuro

        #ajustando pesos sinápticos
        @layers[_L].times do |j|
          (@layers[_L-1] + 1).times do |i|
            @synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
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
          neuro << soma * dtanh(@I[_L][j])
        end

        @gradient[_L] = neuro

        #ajustando pesos sinápticos
        @layers[_L].times do |j|
          (@number_of_entries + 1).times do |i|
            @synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @learning_rate * @gradient[_L][j] * ts[i]
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
          neuro << soma * dtanh(@I[_L][j])
        end

        @gradient[_L] = neuro

        #ajustando pesos sinápticos
        @layers[_L].times do |j|
          (@layers[_L-1] + 1).times do |i|
            @synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
          end
        end
      end
    end

    #------------------------------------------------------------------------------------------- Obtendo Y último atualizado
    #_L = @layers.size - 1
    #@layers.last.times do |j|
    #  @I[_L][j] = 0
    #  @layers[_L-1].times do |i|
    #    @I[_L][j] += @synaptic_weights[_L][j][i] * @Y[_L-1][j]
    #  end
    #end

    #matriz Y
    #@layers.last.times do |j|
    #  @Y[_L][j] = Math.tanh(@I[_L][j])
    #end
  end
  
  # -------------------------------------------------------------------------------------------- Calculando erro
  soma = 0
  @training_samples.size.times do |k|
    soma += @errors[k]
  end
  @error = soma / @training_samples.size
  
  # -------------------------------------------------------------------------------------------- Contando eras
  @age += 1
end until ((@error - @old_error) <= @precision) || @age > 5000

CSV.open("archives/synaptic_weights.csv", "wb") do |csv|
  @layers.each_with_index do |layer, _L|
    layer.times do |j|
      csv << @synaptic_weights[_L][j]
    end
  end
end

puts "Pesos salvos com sucesso!"