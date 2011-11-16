require 'csv'

@layers = [15, 3]
@number_of_entries = 4

#   0.8622, 0.7101, 0.6236, 0.7894

puts "Entre com os valores separados por (VIRGULA): "
string = name = gets.chomp
@inputs = string.split(",")

@synaptic_weights = []
 
csv = CSV.open('archives/synaptic_weights.csv', 'r', {:col_sep => ',', :converters => :float}).to_a

csv.size.times do |i| 
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
end

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
        soma += @synaptic_weights[_L][j][i] * @inputs[i].to_f
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

@layers[@layers.size-1].times do |j|
  puts @Y[@layers.size-1][j]
end