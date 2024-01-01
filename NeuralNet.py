"""
@author: varsha
The class NeuralNetSolver is responsible for converting Neural Network to z3.
This includes two types of Neural Networks: torch.nn and keras tensorflow
In main (at the end of this file): choose the model to run
For torch nn: (an example is provided)
    provide the number of neurons in input, hidden, and output layers
    provide weights and biases for the above dimensions
    provide bound constraints using AddBoundConstraints function of NeuralNetSolver
for keras tensorflow:
    provide model type in keras sequential class (either 1 or 2 provided)
    to add a new model:
        add a function to train the model
        add constraint in AddConstraint function
"""

import numpy as np
import torch
import torch.nn as nn
from z3 import *

import tensorflow.keras as keras
from functools import partial
from sklearn.datasets import load_iris # the iris data set 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def ArrayToTensor(inp_data):
    np_array = np.array(inp_data)
    x_np = torch.from_numpy(np_array)
    return x_np

def ApproxFunZ3(x, limits, limits_y, m, c, n):
    interval = np.linspace(limits[0],limits[1],n);
    currExp = float(limits_y[0]);
    for j in range(n):
        currExp = If(float(interval[j]) <=x, m[j]*x+c[j], currExp);
    if limits_y[1] not in [np.inf]: 
        currExp = If(x >=float(interval[-1]), float(limits_y[1]),currExp);
    else:
        currExp = If(x >=float(interval[-1]),  m[-1]*x+c[-1],currExp);
        
    return currExp;

def Linear_AF(X):
    return X;

def ReLU_AF(X):
    return If(X >= 0, X, 0);

def LeakyReLU_AF(X, negative_slope = 0.01):
    return If(X >= 0, X, negative_slope*X);

def Sigmoid_AF(X):
    m = np.array([0.00337275, 0.0061105 , 0.0110191 , 0.01970494, 0.0347148 ,
           0.05957804, 0.09779728, 0.14937359, 0.2047808 , 0.24276051,
           0.24276051, 0.2047808 , 0.14937359, 0.09779728, 0.05957804,
           0.0347148 , 0.01970494, 0.0110191 , 0.0061105 , 0.00337275]);
    c = np.array([0.02270912, 0.03749296, 0.06105426, 0.09753476, 0.15157027,
           0.22615999, 0.31788617, 0.41072352, 0.47721217, 0.5       ,
           0.5       , 0.52278783, 0.58927648, 0.68211383, 0.77384001,
           0.84842973, 0.90246524, 0.93894574, 0.96250704, 0.97729088]); 
    return  ApproxFunZ3(X, [-6.0,6.0], [0.0,1.0], m, c, 20);

def Tanh_AF(X):
    m = np.array([0.03435273, 0.05599414, 0.09060817, 0.14490914, 0.22745845,
           0.34675794, 0.50578081, 0.69212718, 0.86879398, 0.97967465,
           0.97967465, 0.86879398, 0.69212718, 0.50578081, 0.34675794,
           0.22745845, 0.14490914, 0.09060817, 0.05599414, 0.03435273]);
    c = np.array([-0.90073246, -0.8520393 , -0.78281125, -0.68778454, -0.56396057,
           -0.41483622, -0.25581334, -0.11605357, -0.02772017,  0.        ,
            0.        ,  0.02772017,  0.11605357,  0.25581334,  0.41483622,
            0.56396057,  0.68778454,  0.78281125,  0.8520393 ,  0.90073246]);
    return ApproxFunZ3(X, [-2.5,2.5], [-1.0,1.0], m, c, 20);

def Exp_AF(X):
    m = np.array([0.0932569 ,  0.11974423,  0.15375464,  0.19742487,  0.25349855,
            0.32549858,  0.41794845,  0.53665643,  0.68908049,  0.88479687,
            1.13610167,  1.45878342,  1.87311498,  2.40512725,  3.08824452,
            3.96538445,  5.09165442,  6.53781369,  8.39471895, 10.7790325]);
    c = np.array([  0.31522726,   0.37482375,   0.44284456,   0.51926746,
             0.60337798,   0.69337802,   0.78582789,   0.87485887,
             0.95107091,   1.        ,   1.        ,   0.91932956,
             0.71216378,   0.31315458,  -0.36996269,  -1.46638761,
            -3.15579256,  -5.68657128,  -9.4003818 , -14.76508728]);
    return ApproxFunZ3(X, [-2.5,2.5], [1e-4,np.inf], m, c, 20);

def Softmax_AF(X, otherXs):
    return Exp_AF(X)/Sum([Exp_AF(otherX) for otherX in otherXs]);

class KerasSequential():
    model = None
    def __init__(self, modelType, maxEpochs = 10 ):
        self.modelType = modelType
        if modelType == 1:
            self.model = self.GetIrisNN(maxEpochs)
        elif modelType == 2:
            self.model = self.AndModel(maxEpochs)
     

    def GetIrisNN(self,maxEpochs=100):
        # get the dataset
        iris_dataset = load_iris()
        # split the data
        x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)

        model = keras.Sequential([
            keras.layers.Dense(3, activation='tanh', input_shape=(4,)),
            keras.layers.Dense(3),
            keras.layers.Activation('softmax')
        ])
    
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=maxEpochs) # change epochs
        print("Eval:")
        model.evaluate(x_test,y_test)
        return model
    
    def AndModel(self, maxEpochs = 10):
        def andy(X, epsilon):
            """Boolean AND gate lifted to floating point numbers."""
            return float(all(1-epsilon <= x and x <= 1+epsilon for x in X))
        
        def mk_andy_data(epsilon):
            """Create data for training an AND gate."""
            T = np.linspace(1-epsilon, 1+epsilon, num=100)
            F = np.linspace(0-epsilon, 0+epsilon, num=100)
            D = np.append(T, F)
            X = np.array([ (x1, x2) for x1 in D for x2 in D ])
            Y = np.array(list(map(partial(andy, epsilon=epsilon), X)))
            return (X, Y)
        epsilon = 0.25;
        x_train, y_train = mk_andy_data(epsilon=epsilon)
        model = keras.Sequential([
            keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)),
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=maxEpochs)
        return model
    
    def AddConstraint(self, nnSolver):
        if self.modelType == 1:
            SepalLength = 0
            SepalWidth = 1
            PetalLength = 2
            PetalWidth = 3
            
            Setosa = 0
            Versicolor = 1
            Virginica = 2
            
            def IsSetosa(Y):
              return And(Y[Setosa] > Y[Versicolor], Y[Setosa] > Y[Virginica])
        
            def Between(l, x, u):
              return And([l <= x, x <= u])
    
            X = nnSolver.GetInputVar();
            Y = nnSolver.GetOutputVar(); # this sets the Solver to use the Linear Real Arithmetic
            nnSolver.AddConstraint(And((And(Between(4.0, X[SepalLength], 5.8),
                      Between( 3.5,X[SepalWidth],4.5),
                      Between(0.0, X[PetalLength], 2.0),
                      Between(0.0, X[PetalWidth],1.0))), 
                      (Not(IsSetosa(Y)))))
            #Result Sat for the following constraint
            # nnSolver.AddConstraint(And((And(Between(4.0, X[SepalLength], 5.8),
            #           Between( 2.25,X[SepalWidth],4.5),
            #           Between(0.0, X[PetalLength], 2.0),
            #           Between(0.0, X[PetalWidth],1.0))), 
            #           (Not(IsSetosa(Y)))))
            
        if self.modelType == 2:
            Epsilon = 0.2;
            def Truthy(x):
                return And([1 - Epsilon <= x, x <= 1 + Epsilon])
    
            def Falsey(x):
                return And([0 - Epsilon <= x, x <= 0 + Epsilon])
    
            X = nnSolver.GetInputVar();
            Y = nnSolver.GetOutputVar(); # this sets the Solver to use the Linear Real Arithmetic
    
            nnSolver.AddConstraint(And([Truthy(X[0]), Truthy(X[1]), Not(Y[0] > 0.5)]));
            nnSolver.AddConstraint(And([Falsey(X[0]), Truthy(X[1]), Not(Y[0] < 0.5)]));
            nnSolver.AddConstraint(And([Truthy(X[0]), Falsey(X[1]), Not(Y[0] < 0.5)]));
            nnSolver.AddConstraint(And([Falsey(X[0]), Falsey(X[1]), Not(Y[0] < 0.5)]));
    

# Creating a Neural Network class 
class NeuralNet(nn.Module): 
    def __init__(self, input_dim, hidden_Layer_info, output_dim):
        super(NeuralNet, self).__init__()
        self.layers = [];
        self.activationFun = [];
        current_state  = input_dim;
        
        for i in range(len(hidden_Layer_info)):
            layer = nn.Linear(current_state, hidden_Layer_info[i][0])
            nn.init.zeros_(layer.bias)
            nn.init.zeros_(layer.weight)
            self.layers.append(layer)
            self.activationFun.append(hidden_Layer_info[i][1])
            current_state = hidden_Layer_info[i][0]
            
        out_layer = nn.Linear(current_state, output_dim)
        nn.init.zeros_(out_layer.bias)
        nn.init.zeros_(out_layer.weight)
        self.layers.append(out_layer)
        self.activationFun.append(None)
                
    def weight_update(self, weight_info):
        wgtInfoCtr = 0;
        with torch.no_grad():
            for layerId in range(len(self.layers)):
                for outId in range(model.layers[layerId].out_features):                
                    self.layers[layerId].weight[outId] = ArrayToTensor(weight_info[wgtInfoCtr]);
                    wgtInfoCtr += 1
     
    def bias_update(self, bias_info):
        biasInfoCtr = 0;
        with torch.no_grad():
            for layerId in range(len(self.layers)):
                for outId in range(model.layers[layerId].out_features):                
                    self.layers[layerId].bias[outId] = ArrayToTensor(bias_info[biasInfoCtr]);
                    biasInfoCtr += 1

class NeuralNetSolver():
    def __init__(self, model):
        self.NN = model    
        self.var = []
        self.X = []
        self.Weights = []
        self.Biases = []
        self.AF = []
        if (isinstance(model, keras.Model)):
            for l in range(len(model.layers)):
                layer = model.layers[l]
                params = layer.get_weights()
                # print(params)
        
                # Ignore layers without weights
                if len(params) <= 0: continue
        
                config = layer.get_config()
                rows, cols = params[0].shape
                weights = params[0].T
                biases = params[1]
                if len(self.X) == 0:
                    self.X.append([ Real ('X%s_%s' % (i,0)) for i in range (rows) ])
                self.X.append([ Real ('X%s_%s' % (i,len(self.X))) for i in range (cols) ])
                self.Weights.append(weights);
                self.Biases.append(biases);
                self.AF.append(config['activation']);
        elif (isinstance(model, nn.Module)):
            strToAf = {nn.ReLU:'relu', nn.Sigmoid:'sigmoid', nn.Tanh:'tanh', nn.LeakyReLU:'leakyrelu',nn.Softmax:'softmax',None:None};
            self.X.append([ Real ('X%s_%s' % (i,0)) for i in range (self.NN.layers[0].in_features) ])
            for lId in range(len(self.NN.layers)):
                self.X.append([ Real ('X%s_%s' % (i,lId+1)) for i in range (self.NN.layers[lId].out_features) ])
                thisLayer = self.NN.layers[lId];
                weights = thisLayer.weight.detach().numpy();
                biases = thisLayer.bias.detach().numpy();
                self.Weights.append(weights);
                self.Biases.append(biases);
                self.AF.append(strToAf[self.NN.activationFun[lId]]);
        self.SolverSet = [Solver()]
        
    def CreateNeuralNetConstraints(self, solverId):
        afDict = {'relu':ReLU_AF, 'sigmoid':Sigmoid_AF, 'tanh':Tanh_AF, 'leakyrelu':LeakyReLU_AF, 'softmax':Softmax_AF};
        if len(self.Weights) == 0:
            return False
        for lId in range(len(self.Weights)):
            #thisLayer = self.NN.layers[lId];
            weights = self.Weights[lId];
            biases = self.Biases[lId];
            ins = len(self.X[lId]);
            outs = len(self.X[lId+1]);
            P = [Real ('P%s_%s' % (i,lId)) for i in range (outs)]
            self.SolverSet[solverId].add([P[i] == Sum([float(weights[i,j])*self.X[lId][j] for j in range(ins)]) + float(biases[i]) for i in range(outs)]);
            
            if self.AF[lId] in afDict:
                if (self.AF[lId] == 'softmax'):
                    self.SolverSet[solverId].add([self.X[lId+1][i] == afDict[self.AF[lId]](P[i], P) for i in range (outs) ]);
                else:
                    self.SolverSet[solverId].add([self.X[lId+1][i] == afDict[self.AF[lId]](P[i]) for i in range (outs) ]);
            else:
                self.SolverSet[solverId].add([self.X[lId+1][i] == P[i] for i in range (outs) ]);
        else:
            return True
        
    def Solve(self, printFinalModal = False):
        for solverId in range(len(self.SolverSet)):
            if not self.CreateNeuralNetConstraints(solverId):
                print('Neural Network constraint application failed; Check input NN')
                continue
            if printFinalModal: print(self.SolverSet[solverId])
            res = self.SolverSet[solverId].check()
            if res == sat :
                print(sat)
                mdl = self.SolverSet[solverId].model ()
                for key in mdl:
                    if key.name()[0] == 'X' and (key.name()[-1] in ['0',str(len(self.Weights))]):
                        print(key,str(mdl[key].as_decimal(4)))

            else :
                print (res)
    
    def AddBoundConstraints(self, lowerBound, upperBound, isInputVar, isOrType = False, isNot = False, equality = True, toSolverSet = 0):
        lenLb = len(lowerBound)
        lenUb = len(upperBound)
        lenInVar = len(isInputVar)
        if not (lenInVar == lenUb and lenUb == lenInVar):
            print("Lengths of first three arguments must be same")
            print('No constraints added.')
            return False
        constList = []
        for varId in range(lenInVar):
            index = 0 if isInputVar[varId] else -1
            if (len(lowerBound[varId])!= len(self.X[index]) or len(upperBound[varId])!= len(self.X[index])):
                print('length of lower and upper bound not equal to length of' + str(index) + ' layer.')
                print('No constraints added for '+ str(index))
                continue
            for i in range(len(self.X[index])):
                if lowerBound[varId][i] != None:
                    if equality == True:
                        constList.append(self.X[index][i] >= lowerBound[varId][i]);
                    else:
                        constList.append(self.X[index][i] > lowerBound[varId][i]);
                if upperBound[varId][i] != None:
                    if equality == True:
                        constList.append(self.X[index][i] <= upperBound[varId][i])
                    else:
                        constList.append(self.X[index][i] < upperBound[varId][i]);
        if isOrType: 
            thisConst = Or(constList)
        else:
            thisConst = And(constList)
        self.SolverSet[toSolverSet].add(Not(thisConst) if isNot else thisConst);
    
    def GetVar(self, index):
        if index<len(self.X):
            return self.X[index]
        else:
            print('index exceeds num of layers')
    def GetInputVar(self):
        return self.GetVar(0);
    def GetOutputVar(self):
        return self.GetVar(len(self.X)-1);
    def AddConstraint(self, z3Const,toSolverSet = 0):
        self.SolverSet[toSolverSet].add(z3Const);
    
                         
if __name__ == '__main__':   
    nnToVerify = 2
    if nnToVerify == 1:
        #Example 1: use torch NN
        input_dim = 2            #Num of input neurons
        hidden_dim = 3
        output_dim = 2           #Num of output neurons
        neurons_each_layer = [input_dim, hidden_dim, output_dim]
        hidden_Layer_info = [[3, nn.ReLU]]
        #weight_info must be a list of lists of size hidden_dim + output_dim
        #first hidden_dim element must be lists of size input_dim
        #last output_dim element must be lists of the size hidden_dim
        weight_info = [[2 ,2] ,[3 ,2] ,[6 , -0.5] ,[2 ,3 ,1] ,[ -1 , -4 ,6]]
        #bias_info must be scalars of size hidden_dim + output_dim
        bias_info = [ -1 ,1 , -4 , -6 ,-8]
    
        model = NeuralNet(input_dim, hidden_Layer_info, output_dim)
        model.weight_update(weight_info)
        model.bias_update(bias_info)
    
        nnSolver = NeuralNetSolver(model);
        #Add bound constraints: arguments are as follows 
        #   lower bound
        #   upper bound
        #   is applied on input variable (otherwise output variable)
        #   is Or constraint on the bounds (otherwise And)
        #   use Not on the applied constraint (typically used for output variable)
        #   includes equality (otherwise strict inequality)
        #   apply to a specific solver
        nnSolver.AddBoundConstraints([[0.3,0.1]], [[0.9,0.7]], [True], False, False, True);
        nnSolver.AddBoundConstraints([[0,None]], [[None,0]], [False], False, True, False);

    elif nnToVerify == 2:
        #Example 2: using keras NN: change 
        #kerasType: 1=> Iris dataset, 2=> And Gate
        kerasType = 1;
        kerasObj = KerasSequential(kerasType, 700);
        kerasModel = kerasObj.model;
        nnSolver = NeuralNetSolver(kerasModel);
        kerasObj.AddConstraint(nnSolver);
        
    #Solve: set true to print the final model generated by nnSolver
    nnSolver.Solve(False)
