%%
% -> input_x is the training dataset, rows must be instances and columns must 
%       be features;
% hidden
function [ weights ] = rbf_sin ( input_x, num_hidden_neurons, center_learning_ratio, epochs, weight_learning_ratio )
    [num_input_neurons, num_instances] = size(input_x);
    
    x_normalized = normalize_data( input_x );
    centers = compute_centers( x_normalized, num_hidden_neurons, center_learning_ratio );
    sigma = compute_sigma( centers );
    [weights, error_m] =compute_weights( x_normalized, centers, sigma, 1, epochs, 0.005, eta_learning_rate );


    
end %function rbf_sin

function [out_hidden_neurons, weights] = compute_weights( x, centers, sigma, num_output_neurons, epochs, expected_error, weight_learning_ratio )
    num_instances = size( x, 1 );
    num_hidden_neurons = size( center, 1 );
    out_hidden_neurons = zeros( num_instances, num_hidden_neurons ); % rows: input_x; columns: center
    for i = 1 : num_instances
        mi = euclidian_distance( x(i), centers ); %vetor com as dist�ncias de x(i) a cada um dos centros
        out_hidden_neurons(i,:) = exp( (-1 * mi^2 ) ./ (2*sigma^2) ); % sa�da de cada neur�nio oculto para x(i)
    end %for

    %% Atualizacao dos pesos
    expected_error = 0.005;
    current_epoch = 1;
    net_output = zeros( num_instances, num_output_neurons );    
    weights = rand( num_output_neurons, num_hidden_neurons ); %inicializa os pesos aleatoriamente
    k = 1;

    mean_square_error = zeros( epochs );
    error_m = 99999999;
    while error_m > expected_error && current_epoch < epochs
        for i = 1 : num_instances
            net_output(i,:) = net_output(i,:) * weights ;
        end %for
        net_output = k .* net_output;
        error_m = abs(sin_values - net_output);
        delta_weight = eta_learning_rate * net_output * error';
        weights = weights + delta_weight;

        mean_square_error(current_epoch) = sum( error_m .^2 )/size( error_m, 2 );
        current_epoch = current_epoch + 1;
    end %while

end %function

%% centers_computation
% x must be normalized. rows must be attributes and columns must be
% instances
function [neuron_centers, quantization_error] = compute_centers( x, num_hidden_neurons, eta_learning_rate )
    [num_input_neurons, num_instances] = size(x);
    
    %initialize centers
    neuron_centers = rand( num_hidden_neurons, num_input_neurons ) - 0.5;
    
    min_distances = zeros( 1,num_instances );
    iterations = 1000;
    
    quantization_error = 999998;
    last_quantization_error = quantization_error + 1;
    while last_quantization_error > quantization_error
        for j = 1 : num_instances
            % find closest center, algorithm "win takes all"
            distances = euclidian_distance( x(j), neuron_centers );
            [min_distance, closest_center_index] = min( distances );
            min_distances(j) = min_distance;
            % update closest center
            neuron_centers( closest_center_index ) = neuron_centers( closest_center_index ) + ...
                eta_learning_rate * ( x(j) - neuron_centers(closest_center_index) );
        end %for
        % quantization_error computation
        last_quantization_error = quantization_error;
        quantization_error = sum( min_distances .^ 2 ) / num_instances;
    end %while
end %function compute_center

function [ normalized_x ] = normalize_data( x )
    num_features = size( x, 1 );
    normalized_x = x;
    for i = 1 : num_features
        max_x = max( x( i, : ) );
        min_x = min( x( i, : ) );
        normalized_x( i, : ) = ( 2 * x( i, : ) - max_x - min_x ) / ( max_x - min_x );
    end
end

%% OBS: For this application 
% x is a row matrix (each column means an attribute value)
% y is a matrix where each row is an element and each means an attribute 
% Returns a column matrix of distances
function [distance] = euclidian_distance( x, y )
    col_dim = 2;
    distance = sqrt( sum( ( x - y ).^ 2, col_dim ) );
end

%%
%{
sigma is computed for each center (hidden neuron)
-> neuron_centers is a matrix where 
    rows number are the same as hidden_neurons number and
    columns number are the same as input_neurons number
%} 

function [sigma] = compute_sigma( neuron_centers )
    %initialize sigma
    num_neuron_centers = size( 1, neuron_centers );
    sigma = zeros( num_neuron_centers );
    for i = 1 : num_neuron_centers
        aux_neuron_centers = neuron_centers;
        aux_neuron_centers(i) = []; % remove the i element
        sigma(i) = min( euclidian_distance( neuron_centers(i), aux_neuron_centers ) )/2;
    end %for
end

function [result] = gaussian_rbf( x, center, std_dev )
    result = exp( -1 / ( 2 * s .^ 2 ) * ( x - center ) );
end