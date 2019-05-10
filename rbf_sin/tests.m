
neuron_centers = [0.797951090691078;-0.299999422022067;0.217175400305654;-0.778594618844669];
neuron_centers = [1,2;3,4];
num_neuron_centers = size( neuron_centers, 1 );
    sigma = zeros( size( neuron_centers ) );
    for i = 1 : num_neuron_centers
        aux_neuron_centers = neuron_centers;
        aux_neuron_centers(i) = []; % remove the i element
        sigma(i) = min( euclidian_distance( neuron_centers(i), aux_neuron_centers ) )/2;
    end %for
    
    
function [distance] = euclidian_distance( x, y )
    col_dim = 2;
    distance = sqrt( sum( ( x - y ).^ 2, col_dim ) );
end