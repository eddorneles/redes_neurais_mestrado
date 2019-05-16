function[] = start_run()
    load('dataset_training.mat');
    eta_center_learning = 0.05;
    eta_weight_learning_ratio = 0.05;
    
    epochs = 1000;
    a = [12,1,2.5,7.32;6,5,4,7];
    weights = rbf_sin( a, 5, ...
        eta_center_learning, epochs, eta_weight_learning_ratio );
    
end