function [neurons_centers] = rbf_sin_training ( input_x, hidden_neurons, learning_ratio_eta, epochs )
    input_neurons = 1;
    
    epochs = 1000;
    learning_ratio_eta = 0.05;
    
    input = normalize( input_x )
   
    neuron_centers = rand( hidden, input_neurons ) - 0.5;
    
    % centers computation
    min_distances = zeros( input_size );
    for i = input_size
        % closest center to input(i) computation
        [diff_value, closest_center_index] = ... 
            min( input_x(i) - neuron_centers  )

        %atualizar centro
        neuron_centers( closest_center_index ) = ...
            neuron_centers( closest_center_index ) + ...
            eta * ( input_x(i) - neuron_centers(closest_center_index) );
        %neuron_centers = neuron_centers + eta
        min_distances(i) = diff_value;
    end %atualização dos centros
    
end

function [ normalized_x ] = normalize( x )
    min_x = min(x);
    max_x = max(x);
    normalized_x = ( 2 * input_x - max_x - min_x ) / max_x - min_x;
end

function [ result ] = gaussian( input_x, neuron_centers )
    sigma_divisor = 2 * sigma_computation( neuron_centers ) .^ 2;
    %compute sigma and get the neuron
    input_x_size = size( input_x, 2 );
    neuron_centers_size = size( neuron_centers, 2 );
    mi_distances = size( 1, neuron_centers );
    for i = 1 : neuron_centers
        diff = abs( input_x - neuron_centers(i) );
        mi_distances(i) = pdist( [ diff ; zeros( 1, input_x_size ) ] );
    end %for
    exp( - squared_distance /  );
end

function [sigma] = sigma_computation( neuron_centers )
    sigma = zeros( 1, neuron_centers );
    neuron_centers_size = size( neuron_centers, 2 );
    for i = 1 : neuron_centers_size
        aux_neuron_centers = neuron_centers;
        aux_neuron_centers(i) = []; % remove the i element
        sigma(i) = min( abs( neuron_centers(i) - aux_neuron_centers ) )/2;
    end %for
end