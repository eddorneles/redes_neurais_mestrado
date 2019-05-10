%%
% -> input_x is the training dataset, rows must be instances and columns must 
%       be features;
% hidden
function [ weights ] = rbf_sin ( input_x, num_hidden_neurons, center_learning_ratio, epochs, weight_learning_ratio )
    [num_input_neurons, num_instances] = size(input_x);
    
    x_normalized = normalize_data( input_x );
    centers = compute_centers( x_normalized, num_hidden_neurons, center_learning_ratio );
    sigma = compute_sigma( centers );
    [weights, error_m] = compute_weights( x_normalized, centers, sigma, 1, epochs, 0.005, weight_learning_ratio );


    
end %function rbf_sin

function [out_hidden_neurons, weights] = compute_weights( x, centers, sigma, num_output_neurons, epochs, expected_error, weight_learning_ratio )
    num_instances = size( x, 2 );
    num_hidden_neurons = size( centers, 2 );
    %% calculo da saída de cada neuronio escondido para cada padrao de entrada
    out_hidden_neurons = zeros( num_hidden_neurons, num_instances ); % rows: input_x; columns: hidden_neuron
    for j = 1 : num_instances
        mi = euclidian_distance( x(:,j), centers ); %vetor com as distancias de x(i) a cada um dos centros
        out_hidden_neurons(:,j) = exp( ( -1 * mi.^2 ) ./ (2*sigma.^2) )'; % saida de cada neuronio oculto para x(j)
    end %for

    %% Atualizacao dos pesos TERMINAR A CORRECAO 
    expected_error = 0.005;
    current_epoch = 1;
    net_output = ones( num_output_neurons, num_instances );
    weights = rand( num_output_neurons, num_hidden_neurons ); %inicializa os pesos aleatoriamente
    k = 1;

    mean_square_error = zeros( epochs );
    error_m = 99999999;
    while error_m > expected_error && current_epoch < epochs
        for j = 1 : num_instances
            net_output(j,:) = net_output(j,:) * weights;
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
    neuron_centers = rand( num_input_neurons, num_hidden_neurons ) - 0.5;
    
    min_distances = zeros( 1, num_instances );
    iterations = 1000; % max iterations while updating centers
    
    quantization_error = 999998;
    last_quantization_error = quantization_error + 1;
    while last_quantization_error > quantization_error
        for j = 1 : num_instances
            % find closest center, algorithm "winner takes all"
            distances = euclidian_distance( x(:,j), neuron_centers );
            [min_distance, closest_center_index] = min( distances );
            min_distances(j) = min_distance;
            % update closest center
            neuron_centers( :, closest_center_index ) = neuron_centers( :, closest_center_index ) + ...
                eta_learning_rate .* ( x(:,j) - neuron_centers( :, closest_center_index) );
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
% x and y must have the same columns and rows number.
% each row is an instance, whereas each column is a feature
function [distance] = euclidian_distance( x, y )
    row_dim = 1;
    distance = sqrt( sum( ( x - y ).^ 2, row_dim ) ); % sum elements of the same column
end

%%
%{
sigma is computed for each center (hidden neuron)
-> neuron_centers is a matrix where rows number are the same as input 
    neurons number and columns number are the same as centers number (hidden neurons)
%} 
function [sigma] = compute_sigma( neuron_centers )
    %initialize sigma
    num_neuron_centers = size( neuron_centers, 2 );
    sigma = zeros( 1, num_neuron_centers );
    for j = 1 : num_neuron_centers
        aux_neuron_centers = neuron_centers;
        aux_neuron_centers(:,j) = []; % remove the j element
        sigma(j) = min( euclidian_distance( neuron_centers(:,j), aux_neuron_centers ) )/2;
    end %for
end

function [result] = gaussian_rbf( x, center, std_dev )
    result = exp( -1 / ( 2 * s .^ 2 ) * ( x - center ) );
end