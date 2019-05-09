function[] = start_run()
    load('dataset_training.mat');
    eta_center_learning = 0.05;
    eta_weight_learning_ratio = 0.05;
    
    epochs = 1000;

    weights = rbf_sin( dataset_training, 4, ...
        eta_center_learning, epochs, eta_weight_learning_ratio );
    
end