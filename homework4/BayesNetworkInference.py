from BayesianNetworks import inference, readFactorTablefromData
import numpy as np
import pandas as pd
from copy import deepcopy

class riskFactorNetwork:
    def __init__(self):
        self._riskFactorNet = pd.read_csv('RiskFactorsData.csv')
        self._connection = {'income':['income'], 'smoke':['smoke', 'income'], 'exercise':['exercise', 'income'], 
                           'bmi':['bmi', 'income'], 'bp':['bp', 'exercise', 'income', 'smoke'],
                           'cholesterol':['cholesterol', 'exercise', 'income', 'smoke'],'diabetes':['diabetes', 'bmi'], 
                           'stroke':['stroke', 'bmi', 'bp', 'cholesterol'], 'attack':['attack', 'bmi', 'bp', 'cholesterol'], 
                           'angina':['angina', 'bmi', 'bp', 'cholesterol']}
        self._factor_dict = {}
        for node,parent_nodes in self._connection.items():
            self._factor_dict[node] = readFactorTablefromData(self._riskFactorNet, parent_nodes)

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
    print('\nIf I have bad habits:')
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
    for i in range(1,9):
        probs[str(i)]=[]
        for outcome in outcomes:
            factor = Network.infer(outcome, ['income'], [i])
            probs[str(i)].append(float(factor[factor[outcome]==1]['probs']))
    df = pd.DataFrame(probs)
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
    print(Network3.infer('stroke', ['diabetes'], [1]))
    print(Network3.infer('stroke', ['diabetes'], [3]))

if __name__=='__main__':
    Network = riskFactorNetwork()
    outcomes = ['diabetes', 'stroke', 'attack', 'angina']
    question1(Network)
    question2(Network, outcomes)
    question3(Network, outcomes)
    Network2 = question4(Network, outcomes)
    question5(Network2, outcomes)
    
    
