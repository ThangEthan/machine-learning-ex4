input_layer_size  = 2;  % 20x20 Input Images of Digits
 hidden_layer_size = 4;   % 25 hidden units
 num_labels = 1;          % 10 labels, from 1 to 10   
X = [0 0; 1 1; 0 1; 1 0];
y = [0; 0; 1; 1];
init_Theta1 = randInitializeWeights(2, 4);
init_Theta2 = randInitializeWeights(4,  1);
initial_nn_params = [init_Theta1(:) ; init_Theta2(:)];
costFunction = @(p) nnCostFunction(p, 2, 4, 1, X, y, 1);
options = optimset('MaxIter', 1);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                  hidden_layer_size, (input_layer_size + 1));
 
 Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):  end), ...
                  num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);
 
 fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
