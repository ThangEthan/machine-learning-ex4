function g = relu(z)
 %SIGMOID Compute sigmoid functoon
 %   J = SIGMOID(z) computes the sigmoid of z.
 
 g = max(0.01 * z, z); 
 end

