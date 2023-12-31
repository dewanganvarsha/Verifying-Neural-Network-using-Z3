# Verifying-Neural-Network-using-Z3
In this project, I implemented a Neural Network using PyTorch and specified the properties of it in the language understandable by Z3. The project focuses on constraint-based verification, handling non-linear activation functions, translating the neural network model into Z3, and verifying the properties using Z3. I also demonstrate the extension for Keras Tensorflow Neural Network by using an AND gate and Iris dataset.

The class **NeuralNetSolver** is responsible for converting **Neural Network** to **z3**.
This includes two types of Neural Networks: torch.nn and keras tensorflow
In main (at the end of NeuralNet.py file): choose the model to run
- For torch nn: (an example is provided)
    - provide the number of neurons in input, hidden, and output layers
    - provide weights and biases for the above dimensions
    - provide bound constraints using AddBoundConstraints function of NeuralNetSolver
- for keras tensorflow:
    - provide model type in keras sequential class (either 1 or 2 provided)
    - to add a new model:
        - add a function to train the model
        - add constraint in AddConstraint function

The activation functions are approximated using piecewise linear interpolation computed in LinApprox.py:
![activation_function_image](./Images/activationFunctions.png)
The exponential function is used for Softmax activation function.

The result for constraint on Setosa shown by box on Iris dataset which produces unsat is:
![activation_function_image](./Images/Constraint_Unsat.png)
This is due to the inclusion of non-Setosa in the boxes. This constraint is commented in the NeuralNet.py file.

The modified constraint to get sat result is:
![activation_function_image](./Images/Constraint_Sat.png)
