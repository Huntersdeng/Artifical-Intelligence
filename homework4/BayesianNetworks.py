import numpy as np
import pandas as pd
from functools import reduce
from copy import deepcopy
## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed 
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs
    return factorTable

## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)
   
    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i,'probs'] = sum(a == (i+1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j,'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable

def joinFactors(Factor1, Factor2):
    # your code
    if (not Factor1.empty) and (not Factor2.empty):
        overlap = list(set(Factor1.columns).intersection(set(Factor2.columns)))
        overlap.remove('probs')
        mask_Factor1 = pd.DataFrame.copy(Factor1)
        mask_Factor2 = pd.DataFrame.copy(Factor2)
        mask_Factor1['a']=1
        mask_Factor2['a']=1
        overlap.append('a')
        Factor = pd.merge(mask_Factor1, mask_Factor2, how='outer', on=overlap)
        Factor['probs_x'] *= Factor['probs_y']
        Factor = Factor.rename(columns={'probs_x':'probs'}).drop(columns=['probs_y','a'])
        return Factor
    else:
        if Factor1.empty:
            return Factor2
        if Factor2.empty:
            return Factor1


## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    # your code 
    if type(hiddenVar)!=list:
        hiddenVar = [hiddenVar]
    fT = pd.DataFrame.copy(factorTable)
    for var in hiddenVar:
        try:
            fT = fT.drop(columns=var)
        except KeyError:
            pass
    varnames = list(fT.columns)
    varnames.remove('probs')
    if len(varnames)==0:
        return pd.DataFrame(columns=['probs'])
    fT = fT[fT.columns].groupby(varnames, as_index=False).sum()
    return fT

## Marginalize a list of variables 
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
##
## Should return a Bayesian network containing a list of factor tables that results
## when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    # your code 
    marginalized_bayesNet = []
    for factorTable in bayesNet:
        fT = marginalizeFactor(factorTable, hiddenVar)
        if fT.empty:
            marginalized_bayesNet.append(fT)
    return marginalized_bayesNet

## Update BayesNet for a set of evidence variables
## bayesnet: a list of factor and factor tables in dataframe format
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals):
    # your code
    if len(evidenceVars)!=0:
        if type(evidenceVars)!=list:
            evidenceVars = [evidenceVars]
        if type(evidenceVals)!=list:
            evidenceVals = [evidenceVals]
        updated_bayesnet = bayesnet
        for var,val in zip(evidenceVars, evidenceVals):
            mask_bayesnet = updated_bayesnet.copy()
            updated_bayesnet = []
            for net in mask_bayesnet:
                try:
                    net = net[net[var]==int(val)]
                    updated_bayesnet.append(net)
                except KeyError:
                    updated_bayesnet.append(net)
        return updated_bayesnet
    else:
        return bayesnet


## Run inference on a Bayesian network
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using 
## join and marginalization of the sets of variables. 
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesnet, hiddenVar, evidenceVars, evidenceVals):
    # your code
    Vars = []
    for fT in bayesnet:
        Vars += list(fT.columns)
    Vars = set(Vars)
    Vars.remove('probs')
    infer_bayesnet = evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals)
    for var in Vars:
        mask_bayesnet = infer_bayesnet.copy()
        infer_bayesnet = []
        factor = pd.DataFrame(columns=['probs'])
        for fT in mask_bayesnet:
            if var in fT.columns:
                factor = joinFactors(factor, fT)
            else:
                infer_bayesnet.append(fT)
        if var in hiddenVar:
            factor = marginalizeFactor(factor, var)
        infer_bayesnet.append(factor)
    return normalize(factor)

def normalize(Factor):
    s = reduce(lambda x,y:x+y,list(Factor['probs']))
    Factor['probs']/=s
    return Factor

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