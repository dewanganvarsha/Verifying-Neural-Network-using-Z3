# Verifying-Neural-Network-using-Z3
In this project, I implemented a Neural Network using PyTorch and specified the properties of it in the language understandable by Z3. The project focuses on constraint-based verification, handling non-linear activation functions, translating the neural network model into Z3, and verifying the properties using Z3.

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
