
input_x = [1,3,5,6,8,10];
normalized = normalize( input_x );
input_size = size( input_x ,1 );

%neuron_centers = rand( 6, 1 ) - 0.5
neuron_centers = [-0.3424,    0.4706,    0.4572,   -0.0146,    0.3003,   -0.3581];
eta = 0.05;
min_distances = zeros( input_size );
for i = input_size
    % calculo do centro mais proximo da entrada x_i
    [diff_value, closest_center_index] = min( sqrt( (input_x(i) - neuron_centers ).^2 ) )
    
    %atualizar centro
    neuron_centers( closest_center_index ) = neuron_centers( closest_center_index ) + ...
        eta * ( input_x(i) - neuron_centers(closest_center_index) );
    %neuron_centers = neuron_centers + eta
    min_distances(i) = diff_value;
end
%% cálculo do erro de quantização
quantization_error = sum(min_distances) / input_size;