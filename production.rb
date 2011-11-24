require 'csv'

@layers = [3, 1]
@number_of_entries = 2

#   0.8622, 0.7101, 0.6236, 0.7894

#puts "Entre com os valores separados por (VIRGULA): "
#string = gets.chomp
#inputs = string.split(",")

@synaptic_weights = []

# g(u) = 1/(1+e^(-bu)) Função Logistica
def funcao_sigmoid(valor)
    return 1/(1 + (Math::E ** (-1 * 0.5 * valor)))
end
 
csv = CSV.open('archives/synaptic_weights.csv', 'r', {:col_sep => ',', :converters => :float}).to_a


@layers.each_with_index do |layer, _L|
  #puts "layer: "+_L.to_s
  neuro = []
  soma = 0
  (0..(_L-1)).each do |t|
    soma += @layers[t]
  end
  layer.times do |j|
    #puts "neuronio: "+j.to_s
    entry = []
    entry = csv[soma+j]
    neuro << entry
  end
  @synaptic_weights << neuro
end

#puts @synaptic_weights.map{|x| x.map{ |y| y.join(" | ")}}.to_s

@tabela = []

CSV.open('archives/tabela_xor.csv', 'r', {:col_sep => ',', :converters => :float}) do |cvs|
  cvs.each do |row|
    @tabela << [-1, row[0], row[1]]
  end
end

@tabela.each do |inputs|
  #puts inputs.join(" | ")
  @I = []
  @Y = []

  #------------------------------------------------------------------------------------------------ forward
  # Preenchendo as matrizes I e Y
  @layers.each_with_index do |layer, _L|
    if _L == 0 #-------------------------------------------------- _L primeiro
      #matriz I
      neuro = []
      layer.times do |j|
        soma = 0
        (@number_of_entries + 1).times do |i|
          #puts @synaptic_weights[_L][j][i]
          soma += @synaptic_weights[_L][j][i] * inputs[i]
        end
        neuro << soma
      end

      @I << neuro

      #matriz Y
      neuro = []
      neuro << -1
      layer.times do |j|
        #neuro << Math.tanh(@I[_L][j-1])
        neuro << funcao_sigmoid(@I[_L][j])
      end

      @Y << neuro
    elsif _L == (@layers.size - 1) #--------------------------------- _L último
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
      layer.times do |j|
        #neuro << Math.tanh(@I[_L][j])
        neuro << funcao_sigmoid(@I[_L][j])
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
        #neuro << Math.tanh(@I[_L][j-1])
        neuro << funcao_sigmoid(@I[_L][j])
      end

      @Y << neuro
    end
  end

  result = []
  @layers[@layers.size-1].times do |j|
    if @Y[@layers.size-1][j] >= 0.5
      result << 1
    else
      result << 0
    end
  end

  #puts @Y[@layers.size-1].join(" | ")
  puts result.join(" | ")
end