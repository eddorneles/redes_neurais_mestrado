function[] = create_sin_dataset()
    %cria matriz linha de 10000 elementos variando entre 0 e 2*pi
    dataset = 2 * pi * rand( 1, 10000 );
    save( 'dataset.mat');

    %divide o dataset em treino (70%) e teste (30%)
    [dataset_training, dataset_test] = divideblock( dataset, 0.7, 0.3, 0 );
    
    save( 'dataset_training.mat' );
    save( 'dataset_test.mat' );
end %function