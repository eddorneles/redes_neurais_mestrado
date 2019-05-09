
input_x = [1,3,5,6,8,10]';
x_normalized = normalize( input_x )';
input_size = size( input_x ,1 );

%neuron_centers = rand( 6, 1 ) - 0.5
neuron_centers = [-0.3424, 0.4706, 0.4572, -0.0146, 0.3003,   -0.3581]';
eta = 0.05;
min_distances = zeros( input_size );
row_dim = 1; % dimension to be summed
for i = 1 : input_size
    % calculo do centro mais proximo da entrada x_i
    
    %compute the distance between x_normalized(i) and all neuron_centers,
    %returns a row matrix of distance
    
    distance = compute_euclidian_distance( x_normalized(i), neuron_centers );
    [ min_distance, closest_center_index ] = min( distance );

    
    %atualizar centro
    neuron_centers( closest_center_index ) = neuron_centers( closest_center_index ) + ...
        eta * ( x_normalized(i) - neuron_centers(closest_center_index) );
    %neuron_centers = neuron_centers + eta
    min_distances(i) = diff_value;
end
%% cálculo do erro de quantização
quantization_error = sum(min_distances) / input_size;


%% OBS: For this application x is a row matrix (each element means an attribute) 
% Returns a row matrix of distances
function [distance] = compute_euclidian_distance( x, y )
    row_dim = 1;
    distance = sqrt( sum( ( x - y ).^ 2, row_dim ) );
end


