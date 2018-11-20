#!/usr/bin/env python3

from BayesianNetworks import *
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

#############################
## Example Tests from Bishop Pattern recognition textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors 
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)
marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, 'fuel', '1')
evidenceUpdateNet(carNet, ['fuel', 'battery'], ['1', '0'])

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.
marginalizeNetworkVariables(carNet, 'battery') ## this returns back a list
marginalizeNetworkVariables(carNet, 'fuel') ## this returns back a list
marginalizeNetworkVariables(carNet, ['battery', 'fuel'])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
## you need to create more factor tables

risk_net = [income, smoke, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise'})
obsVars  = ['smoke', 'exercise']
obsVals  = [1, 2]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)


### Please write your own test scrip similar to  the previous example 
###########################################################################
#HW4 test scrripts start from here
###########################################################################
## define a class to create the bayesnet
class riskFactorNetwork:
    def __init__(self):
        self._riskFactorNet = pd.read_csv('RiskFactorsData.csv')
        self._connection = {'income':['income'], 'smoke':['smoke', 'income'], 'exercise':['exercise', 'income'], 
                           'bmi':['bmi', 'income', 'exercise'], 'bp':['bp', 'exercise', 'income', 'smoke'],
                           'cholesterol':['cholesterol', 'exercise', 'income', 'smoke'],'diabetes':['diabetes', 'bmi'], 
                           'stroke':['stroke', 'bmi', 'bp', 'cholesterol'], 'attack':['attack', 'bmi', 'bp', 'cholesterol'], 
                           'angina':['angina', 'bmi', 'bp', 'cholesterol']}
        self._factor_dict = {}
        for node,parent_nodes in self._connection.items():
            self._factor_dict[node] = readFactorTablefromData(self._riskFactorNet, parent_nodes)

    ## get a list of factor table
    def get_network(self):
        return [factor for factor in self._factor_dict.values()]
        
    def size(self):
        size = 0
        for factor in self._factor_dict.values():
            size += factor.iloc[:,0].size
        return size

    def change_edge(self, parent_nodes, nodes):
        for parent_node in parent_nodes:
            for node in nodes:
                self._connection[node].append(parent_node)
                self._factor_dict[node] = readFactorTablefromData(self._riskFactorNet, self._connection[node])

    def infer(self, outcome, evidenceVars, evidenceVals):
        risk_net = self.get_network()
        hiddenVar = [var for var in self._factor_dict]
        hiddenVar.remove(outcome)
        for var in evidenceVars:
            hiddenVar.remove(var)
        return inference(risk_net, hiddenVar, evidenceVars, evidenceVals)

def question1(Network):
    print('the size of this network is ', Network.size())

def question2(Network, outcomes):
    print('If I have bad habits:')
    for outcome in outcomes:
        print('The probabilities of the ', outcome, ' is\n' ,Network.infer(outcome, ['smoke', 'exercise'], [1,2]))
    print('\nIf I have good habits:')
    for outcome in outcomes:
        print('The probabilities of the ', outcome, ' is\n' ,Network.infer(outcome, ['smoke', 'exercise'], [2,1]))
    print('\nIf I have poor health:')
    for outcome in outcomes:
        print('The probabilities of the ', outcome, ' is\n' ,Network.infer(outcome, ['bp', 'cholesterol', 'bmi'], [1,1,3]))
    print('\nIf I have good health:')
    for outcome in outcomes:
        print('The probabilities of the ', outcome, ' is\n' ,Network.infer(outcome, ['bp', 'cholesterol', 'bmi'], [3,2,2]))

def question3(Network, outcomes):
    probs = {}
    index={0:'diabetes', 1:'stroke', 2:'attack', 3:'angina'}
    color=['green', 'red', 'blue', 'Yellow']
    for i in range(1,9):
        probs[str(i)]=[]
        for outcome in outcomes:
            factor = Network.infer(outcome, ['income'], [i])
            probs[str(i)].append(float(factor[factor[outcome]==1]['probs']))
    df = pd.DataFrame(probs)
    for i in range(4):
        plt.plot(df.iloc[i], color=color[i], label=index[i])
    plt.legend()
    plt.xlabel('income')
    plt.ylabel('health outcomes')
    plt.show()
    df = df.rename(index, axis='index')

    print(df)

def question4(Network, outcomes):
    Network2 = deepcopy(Network)
    Network2.change_edge(['smoke', 'exercise'], outcomes)
    question2(Network2, outcomes)
    return Network2

def question5(Network2, outcomes):
    Network3 = deepcopy(Network2)
    Network3.change_edge(['diabetes'], ['stroke'])
    print(Network2.infer('stroke', ['diabetes'], [1]))
    print(Network2.infer('stroke', ['diabetes'], [3]))
    print('After adding edge from diabetes to stroke')
    print(Network3.infer('stroke', ['diabetes'], [1]))
    print(Network3.infer('stroke', ['diabetes'], [3]))

if __name__=='__main__':
    Network = riskFactorNetwork()
    outcomes = ['diabetes', 'stroke', 'attack', 'angina']
    print('Question1:')
    question1(Network)
    print('Question2:')
    question2(Network, outcomes)
    print('Question3:')
    question3(Network, outcomes)
    print('Question4:')
    Network2 = question4(Network, outcomes)
    print('Question5:')
    question5(Network2, outcomes)


