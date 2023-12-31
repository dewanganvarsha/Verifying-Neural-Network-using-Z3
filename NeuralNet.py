# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:43:21 2023

@author: varsha
"""

#Define the network

import numpy as np
import torch
import torch.nn as nn
from z3 import *

input_dim = 2            #Num of input neurons
num_hidden_layers = 2    #Num of hidden layers 
hidden_dim = 3
output_dim = 2           #Num of output neurons
neurons_each_layer = [input_dim, hidden_dim, output_dim]

def ArrayToTensor(inp_data):
    np_array = np.array(inp_data)
    x_np = torch.from_numpy(np_array)
    return x_np

def DotProduct(data1, data2):
    data1.data2 = torch.dot(torch.tensor(data1), torch.tensor(data2))
    return data1.data2

def ApproxFun(x, limits, limits_y, m, c, n):
    y = np.zeros(x.size)
    for i in range(len(x)):
        if x[i]<=limits[0]:
            y[i]= limits_y[0]
        elif x[i]>=limits[1]:
            y[i]= limits_y[1]
        else:
            interval = np.linspace(limits[0],limits[1],n)
            for j in range(n-1):
                if (interval[j] <=x[i] and x[i] <= interval[j+1])== True:
                    y[i]= m[j]*x[i]+c[j]
                    break
    return y

def ApproxFunScalar(x, limits, limits_y, m, c, n):
    if x <=limits[0]:
        return limits_y[0]
    elif x >=limits[1]:
        return limits_y[1]
    else:
        interval = np.linspace(limits[0],limits[1],n)
        for j in range(n-1):
            if (interval[j] <=x and x <= interval[j+1])== True:
                return m[j]*x+c[j]
    return 0 #Should be NaN or something

def ApproxFunZ3(x, limits, limits_y, m, c, n):
    
    interval = np.linspace(limits[0],limits[1],n);
    currExp = float(limits_y[0]);
    # for j in range(n-1):
    #     currExp = If(And(interval[j] <=x, x <= interval[j+1]), m[j]*x+c[j], currExp);
    for j in range(n-1):
        currExp = If(float(interval[j]) <=x, m[j]*x+c[j], currExp);
    If(x >=float(interval[-1]), float(limits_y[1]),currExp);
        
    return 0 #Should be NaN or something

def ReLU_AF(X):
    return If(X >= 0, X, 0);

def LeakyReLU_AF(X, negative_slope = 0.01):
    return If(X >= 0, X, negative_slope*X);

def Sigmoid_AF(X):   
    limits = [-6,6]
    m = np.array([0.00343145, 0.00641289, 0.01191966, 0.02193238, 0.0396158 ,
            0.06922073, 0.11421441, 0.17174377, 0.22528912, 0.24794295,
            0.22528912, 0.17174377, 0.11421441, 0.06922073, 0.0396158 ,
            0.02193238, 0.01191966, 0.00641289, 0.00343145, 0.        ]);
    c = np.array([0.02306135, 0.03906697, 0.06515164, 0.10625651, 0.16768311,
            0.25182343, 0.35128315, 0.44211899, 0.49284616, 0.5       ,
            0.50715384, 0.55788101, 0.64871685, 0.74817657, 0.83231689,
            0.89374349, 0.93484836, 0.96093303, 0.97693865, 0.        ]); 
    return  ApproxFunZ3(X, limits, [0,1], m, c, 20);

def Tanh_AF(X):
    limits = [-2,2]
    m = np.array([0.08703229, 0.12952133, 0.19050109, 0.27540143, 0.38837541,
            0.5289406 , 0.68719967, 0.84010891, 0.95388885, 0.99632285,
            0.95388885, 0.84010891, 0.68719967, 0.5289406 , 0.38837541,
            0.27540143, 0.19050109, 0.12952133, 0.08703229, 0.        ]);
    c = np.array([-7.89962995e-01, -7.13929980e-01, -6.17646152e-01, -5.01466734e-01,
            -3.70654762e-01, -2.37487743e-01, -1.20875789e-01, -4.03972444e-02,
            -4.46673716e-03, -4.16333634e-17,  4.46673716e-03,  4.03972444e-02,
             1.20875789e-01,  2.37487743e-01,  3.70654762e-01,  5.01466734e-01,
             6.17646152e-01,  7.13929980e-01,  7.89962995e-01,  0.00000000e+00]);
    return ApproxFunZ3(X, limits, [0,1], m, c, 20);

def GetActivationFun(nnFun):
    if nnFun == nn.ReLU():
        return ReLU_AF;

   


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
        self.activationFun.append(nn.ReLU())
                
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
      
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        return x

class NeuralNetSolver():
    def __init__(self, NN):
        self.NN = NN    
        self.var = []
        self.X = []
        self.X.append([ Real ('X%s_%s' % (i,0)) for i in range (self.NN.layers[0].in_features) ])
        for lId in range(len(self.NN.layers)):
            self.X.append([ Real ('X%s_%s' % (i,lId+1)) for i in range (self.NN.layers[lId].out_features) ])
        self.SolverSet = [Solver()]
        
    def CreateNeuralNetConstraints(self, solverId):
        for lId in range(len(self.NN.layers)):
            thisLayer = self.NN.layers[lId];
            weights = thisLayer.weight;
            biases = thisLayer.bias;
            ins = thisLayer.in_features;
            outs = thisLayer.out_features;
            P = [Real ('P%s_%s' % (i,lId)) for i in range (outs)]
            self.SolverSet[solverId].add([P[i] == Sum([float(weights[i,j].detach().numpy())*self.X[lId][j] for j in range(ins)]) + float(biases[i].detach().numpy()) for i in range(outs)]);
            if self.NN.activationFun[lId] == nn.ReLU:
                self.SolverSet[solverId].add([self.X[lId+1][i] == ReLU_AF(P[i]) for i in range (outs) ]);
            elif self.NN.activationFun[lId] == nn.Sigmoid:
                self.SolverSet[solverId].add([self.X[lId+1][i] == Sigmoid_AF(P[i]) for i in range (outs) ]);
            elif self.NN.activationFun[lId] == nn.Tanh:
                self.SolverSet[solverId].add([self.X[lId+1][i] == Tanh_AF(P[i]) for i in range (outs) ]);
            elif self.NN.activationFun[lId] == nn.LeakyReLU:
                self.SolverSet[solverId].add([self.X[lId+1][i] == LeakyReLU_AF(P[i]) for i in range (outs) ]);
            else:
                self.SolverSet[solverId].add([self.X[lId+1][i] == P[i] for i in range (outs) ]);
            
            
            
            # To write in one statement the above three lines
            # self.SolverSet[solverId].add([self.X[lId+1][i] == ReLU_AF(Sum([float(weights[i,j].detach().numpy())*self.X[lId][j] for j in range(ins)]) + float(biases[i].detach().numpy())) for i in range (outs) ]);
                
        
        
        
        # Z = [ Real ('Z%s' % i) for i in range (n)]
        # P = [ Real ('P%s' % i) for i in range (3) ]
        # X = [ Real ('X%s' % i) for i in range (2) ]
        # target = And (Z [3] > 0, Z [4] < 0)
        
        # slv.add([P[i] == Sum([theta[i][j]*X[j] for j in range(2)]) + b[i] for i in range(3)])
        # slv.add ([Z[i] == If(P[i] >= 0, P[i], 0) for i in range (3) ])
        # slv.add ([Z[i] == Sum ([ theta [i][j]*Z[j] for j in range (3) ]) + b[i] for i in range (3 ,5) ])
        
    def Solve(self):
        for solverId in range(len(self.SolverSet)):
            self.CreateNeuralNetConstraints(solverId)
            print(self.SolverSet[solverId])
            res = self.SolverSet[solverId].check()
            if res == sat :
                print(sat)
                mdl = self.SolverSet[solverId].model ()
                print(mdl)
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
            
        
# Linear constraints
# Upper and lower bound
# quadratic 
                         
        

hidden_Layer_info = [[3, nn.LeakyReLU]]
weight_info = [[2 ,2] ,[3 ,2] ,[6 , -0.5] ,[2 ,3 ,1] ,[ -1 , -4 ,6]]
bias_info = [ -1 ,1 , -4 , -6 ,-8]

model = NeuralNet(input_dim, hidden_Layer_info, output_dim)
model.weight_update(weight_info)
model.bias_update(bias_info)
print(*model.layers, sep="\n")
print(*model.activationFun)

nnSolver = NeuralNetSolver(model);
nnSolver.AddBoundConstraints([[0.3,0.1]], [[0.9,0.7]], [True], False, False, True);
#nnSolver.AddBoundConstraints([[0,None]], [[None,0]], [False], True);
nnSolver.AddBoundConstraints([[0,None]], [[None,0]], [False], False, True, False);
nnSolver.Solve()
