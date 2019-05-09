function [] = start_run()
    interval_zero_2pi = 2 * pi * rand( 1, 10000 );
    save( 'dataset.mat');
    
    [training_set, test_set] = divideblock( interval_zero_2pi, 0.7, 0.3, 0 );


end