function [ whi, woh, bias_hi, bias_oh, eav, index ] = trainSinNet( x )
    
    %  Network Architecture
    num_hidden_neurons = 8;
    num_input_ = 1;
    num_output = 1;
    
    eta_learning_ratio = 0.05
    