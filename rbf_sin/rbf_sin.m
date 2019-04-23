function [  ] = rbf_sin ( input_x, hidden_neurons, learning_ratio_eta, epochs )
    input_neurons = 1;
    
    epochs = 1000;
    learning_ratio_eta = 0.05;
    
    input = normalize( input_x )
   
    neuron_centres = rand( hidden, input_neurons ) - 0.5;
    
    % centers computation
    min_distances = zeros( input_size );
    for i = input_size
        % closest center to input(i) computation
        [diff_value, closest_center_index] = ... 
            min( sqrt( (input_x(i) - neuron_centres ).^2 ) )

        %atualizar centro
        neuron_centres( closest_center_index ) = ...
            neuron_centres( closest_center_index ) + ...
            eta * ( input_x(i) - neuron_centres(closest_center_index) );
        %neuron_centers = neuron_centers + eta
        min_distances(i) = diff_value;
    end
end

function [ normalized_x ] = normalize( x )
    min_x = min(x);
    max_x = max(x);
    normalized_x = ( 2 * input_x - max_x - min_x ) / max_x - min_x;
end

function [ result ] = gaussian( input_x, neuron_centres )
    sigma_divisor = 2 * sigma_computation( neuron_centres ) .^ 2;
    %compute sigma and get the neuron
    squared_distance = ( input_x - neuron_centres );
    exp( - squared_distance / );
end

function [sigma] = sigma_computation( neuron_centres )
    sigma = zeros( 1, neuron_centres );
    neuron_centres_size = size( neuron_centres, 2 );
    for i = neuron_centres_size
        aux_neuron_centres = neuron_centres;
        aux_neuron_centres(i) = []; % remove the i element
        sigma(i) = min( abs( neuron_centres(i) - aux_neuron_centres ) )/2;
    end %for
end