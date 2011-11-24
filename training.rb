require 'csv'

@training_samples = []
@desired_output = []

@layers = [3, 1]
@number_of_entries = 2

@learning_rate = 0.1
@precision = 10 ** -8

@momentum = 0.9

@age = 0

@error = 0.0

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dtanh(x)
  return 1.0 - x**2
end

def derivada_sigmoid(val_fun)
    return 0.5 * (funcao_sigmoid(val_fun)) * (1 - funcao_sigmoid(val_fun))
end

# g(u) = 1/(1+e^(-bu)) Função Logistica
def funcao_sigmoid(valor)
    return 1/(1 + (Math::E ** (-1 * 0.5 * valor)))
end

def get_training_and_output
  CSV.open('archives/training_samples_xor.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
    cvs.each do |row|
      @training_samples << [-1, row[0], row[1]]
      @desired_output << [row[2]]
    end
  end
end

get_training_and_output

#puts @training_samples.map{ |x| x.join(" | ")}
#puts @desired_output.map{ |x| x.join(" | ")}

@ws = []
@synaptic_weights = []
@null_synaptic_weights = []

@layers.each_with_index do |layer, _L|
  #puts "layer: "+_L.to_s
  neuro = []
  null_neuro = []
	if _L == 0
    layer.times do |j|
      #puts "neuronio: "+j.to_s
      entry = []
      null_entry = []
      (@number_of_entries + 1).times do |i|
        #puts "entrada: "+i.to_s
        entry << rand
        null_entry << 0
      end
      neuro << entry
      null_neuro << null_entry
    end
	else
    layer.times do |j|
      #puts "neuronio: "+j.to_s
      entry = []
      null_entry = []
      (@layers[_L-1] + 1).times do |i|
        #puts "entrada: "+i.to_s
				entry << rand
        null_entry << 0
			end
      neuro << entry
      null_neuro << null_entry
    end
  end
  @synaptic_weights << neuro
  @null_synaptic_weights << null_neuro
end

@ws << @null_synaptic_weights
@ws << @null_synaptic_weights

#@null_synaptic_weights = Array.new @null_synaptic_weights

#puts @synaptic_weights.map{ |l| l.map{ |x| x.join(" | ")}}
#puts @null_synaptic_weights.map{ |l| l.map{ |x| x.join(" | ")}}
#puts @ws.map{ |y| y.map{|l| l.map{ |x| x.join(" | ")}}}

@error_arch = CSV.open("archives/errors.csv", "wb")

begin
  puts "Entrando na era: "+@age.to_s
  @old_error = @error
  @errors = []

  #puts @synaptic_weights.join(" - ")

  @training_samples.each_with_index do |ts, ti|
    @I = []
    @Y = []
    @gradient = []

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
            #puts soma.to_s+" | "+@synaptic_weights[_L][j][i].to_s+" | "+ts[i].to_s
          end
          neuro << soma
        end

        @I << neuro

        #matriz Y
        neuro = []
        neuro << -1
        layer.times do |j|
          #neuro << Math.tanh(@I[_L][j])
          neuro << funcao_sigmoid(@I[_L][j])
          #puts funcao_sigmoid(@I[_L][j])  
        end

        @Y << neuro
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

        @I << neuro

        #matriz Y
        neuro = []
        layer.times do |j|
          #neuro << Math.tanh(@I[_L][j])
          neuro << funcao_sigmoid(@I[_L][j])
          #puts funcao_sigmoid(@I[_L][j])
        end

        @Y << neuro
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

        @I << neuro

        #matriz Y
        neuro = []
        neuro << -1
        layer.times do |j|
          #neuro << Math.tanh(@I[_L][j])
          neuro << funcao_sigmoid(@I[_L][j])
        end

        @Y << neuro
      end
    end

    #puts @I.map{|x| x.join(" | ")}
    #puts @Y.map{|x| x.join(" | ")} 

    #------------------------------------------------------------------ Calcular erro local 
    soma = 0
    #puts @I[@layers.size-1].join(" , ")
    #puts @Y[@layers.size-1].join(" , ")
    @layers.last.times do |j|
      #puts @desired_output[ti][j].to_s+" - "+@Y[@layers.size-1][j].to_s
      soma += (@desired_output[ti][j] - @Y[@layers.size-1][j]) ** 2
    end
    @errors[ti] = soma / 2

    #------------------------------------------------------------------------------------------------- backward
    #puts @Y[1].join(" - ")
    (@layers.size-1).downto 0 do |_L|
      if _L == (@layers.size - 1) #--------------------------------- _L último
        #gradiente
        neuro = []
        @layers[_L].times do |j|
          #neuro << (@desired_output[ti][j] - @Y[_L][j]) * dtanh(@I[_L][j])
          neuro << (@desired_output[ti][j] - @Y[_L][j]) * derivada_sigmoid(@I[_L][j])
        end

        @gradient[_L] = neuro
        #puts @gradient[_L].join(" | ")
    

        #ajustando pesos sinápticos
        @layers[_L].times do |j|
          (@layers[_L-1] + 1).times do |i|
            @synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f) + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
            #@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
            #puts @synaptic_weights[_L][j][i].to_s+" | "+@learning_rate.to_s+" | "+@gradient[_L][j].to_s+" | "+@Y[_L-1][i].to_s
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
          #neuro << soma * dtanh(@I[_L][j])
          neuro << soma * derivada_sigmoid(@I[_L][j])
        end

        @gradient[_L] = neuro
        #puts @gradient[_L].join(" | ")

        #ajustando pesos sinápticos
        @layers[_L].times do |j|
          (@number_of_entries + 1).times do |i|
            @synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f) + @learning_rate * @gradient[_L][j] * ts[i]
            #@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @learning_rate * @gradient[_L][j] * ts[i]
            #puts @synaptic_weights[_L][j][i].to_s+" | "+@learning_rate.to_s+" | "+@gradient[_L][j].to_s+" | "+ts[i].to_s
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
          #neuro << soma * dtanh(@I[_L][j])
          neuro << soma * derivada_sigmoid(@I[_L][j])
        end

        @gradient[_L] = neuro

        #ajustando pesos sinápticos
        @layers[_L].times do |j|
          (@layers[_L-1] + 1).times do |i|
            @synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @momentum * (@ws[0][_L][j][i].to_f - @ws[1][_L][j][i].to_f) + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
            #@synaptic_weights[_L][j][i] = @synaptic_weights[_L][j][i] + @learning_rate * @gradient[_L][j] * @Y[_L-1][i]
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

    @ws[1] = @ws[0]
    @ws[0] = @synaptic_weights
  end
  
  # -------------------------------------------------------------------------------------------- Calculando erro
  soma = 0
  @training_samples.size.times do |k|
    #puts @errors[k]
    soma += @errors[k]
  end
  @error = soma / @training_samples.size
  
  # -------------------------------------------------------------------------------------------- Contando eras
  @age += 1
  @error_arch << [@error]
  #puts @error.to_s+" | "+@old_error.to_s
  #puts @errors.join(" - ")
  #puts @I.join(" - ")
end until ((@error - @old_error).abs <= @precision) || (@age > 100000)

CSV.open("archives/synaptic_weights.csv", "wb") do |csv|
  @layers.each_with_index do |layer, _L|
    layer.times do |j|
      csv << @synaptic_weights[_L][j]
    end
  end
end

puts "Pesos salvos com sucesso!"