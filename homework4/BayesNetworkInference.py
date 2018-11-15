from BayesianNetworks import *
import numpy as np
import pandas as pd

class riskFactorNetwork:
    def __init__(self, data):
        riskFactorNet = pd.read_csv(data)
        ## create factors
        self.income      = readFactorTablefromData(riskFactorNet, ['income'])
        self.smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
        self.exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
        self.bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
        self.bp          = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'income', 'smoke'])
        self.cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol', 'exercise', 'income', 'smoke'])
        self.diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
        self.stroke      = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol'])
        self.attack      = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol'])
        self.angina      = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol'])
        self.vars = ['income', 'smoke', 'exercise', 'bmi', 'bp', 'stroke', 'attack', 'angina', 'cholesterol', 'diabetes']
        self.risk_net = [self.income, self.smoke, self.exercise, self.bmi, 
                         self.bp, self.cholesterol, self.diabetes, self.stroke,
                         self.attack, self.angina]

    def size(self):
        size = 0
        for factor in self.risk_net:
            size += factor.iloc[:,0].size
        return size

    def infer(self, outcome, evidenceVars, evidenceVals):
        hiddenVar = self.vars.copy()
        hiddenVar.remove(outcome)
        for var in evidenceVars:
            hiddenVar.remove(var)
        return inference(self.risk_net, hiddenVar, evidenceVars, evidenceVals)

if __name__=='__main__':
    Network = riskFactorNetwork('RiskFactorsData.csv')
    print('the size of this network is ', Network.size())
    outcomes = ['diabetes', 'stroke', 'attack', 'angina']
    print('If I have bad habits:\n')
    for outcome in outcomes:
        print('The probabilities of the ', outcome, ' is\n' ,Network.infer(outcome, ['smoke', 'exercise'], [1,2]))
    print('If I have good habits:\n')
    for outcome in outcomes:
        print('The probabilities of the ', outcome, ' is\n' ,Network.infer(outcome, ['smoke', 'exercise'], [2,1]))
    