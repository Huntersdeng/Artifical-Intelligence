
#%%
from BayesianNetworks import *
from BayesNetworkTestScript import *
import numpy as np
import pandas as pd
%pylab inline

#%%
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


#%%
a = pd.merge(smoke, bmi, on='income', how='outer')
a['probs_x'] *= a['probs_y']
a = a.rename(columns={'probs_x':'probs'}).drop(columns=['probs_y','income'])
a


#%%
a = a[a.columns].groupby(['smoke','bmi'], as_index=False).sum()
a


#%%
bmi


#%%
pd.merge(smoke, bmi.drop(columns='probs'), on='income', how='outer')


#%%
def find_index(mask, value):
    return int(np.argwhere(np.all(mask[mask.columns].values==value, axis=1)==True))
smoke



#%%
def func(Factor, mask, elements, dic):
    element = elements.pop()
    outcome = set(Factor[element])
    for i in outcome:
        dic[element]=i
        if len(elements)==0:
            indexes = list(dic.items())
            Factor[Factor[element]==i]['probs']*=func2(mask, indexes)
        else:
            func(Factor[Factor[element]==i], mask[element], elements, dic)


#%%
def func2(mask, indexes):
    index = indexes.pop()
    if len(indexes)==0:
        return mask[mask[index[0]]==index[1]]['prob']
    else:
        func2(mask[mask[index[0]]==index[1]], indexes)


#%%
smoke[smoke['income']==1][smoke['smoke']==1]['probs']*smoke[smoke['income']==1][smoke['smoke']==1]['probs'].values


#%%
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF]


#%%
a = list(BatteryState.columns)
b = list(FuelState.columns)
b.remove('probs')
varnames = a+b
varnames.remove('probs')
l = len(varnames)
data = {}
probs = {}
for i in range(l):
    data[varnames[i]]=[]
probs={}
probs['probs']=[]
width1 = BatteryState.iloc[:,0].size
width2 = FuelState.iloc[:,0].size
for i in range(width1):
    for j in range(width2):
        probs['probs'].append(BatteryState['probs'][i]*FuelState['probs'][j])
        for k in range(l):
            try:
                data[varnames[k]].append(BatteryState[varnames[k]][i])
            except KeyError:
                data[varnames[k]].append(FuelState[varnames[k]][j])
print(probs)
print(data)


#%%
varnames.append('probs')
raw_data = probs.copy()
raw_data.update(data)
df = pd.DataFrame(raw_data, columns=varnames)
df


#%%
joinFactors(BatteryState, FuelState)


#%%
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)


#%%
dic = {'fuel':1, 'battery':1, 'gauge':1}
l1= dic.items()
list(l1)


#%%
joinFactors(BatteryState, FuelState)


#%%
pd.merge(BatteryState, FuelState, on='probs')


#%%
connection = {'income':['income'], 'smoke':['smoke', 'income'], 'exercise':['exercise', 'income'], 
                           'bmi':['bmi', 'income'], 'bp':['bp', 'exercise', 'income', 'smoke'],
                           'cholesterol':['cholesterol', 'exercise', 'income', 'smoke'],'diabetes':['diabetes', 'bmi'], 
                           'stroke':['stroke', 'bmi', 'bp', 'cholesterol'], 'attack':['attack', 'bmi', 'bp', 'cholesterol'], 
                           'angina':['angina', 'bmi', 'bp', 'cholesterol']}


#%%
l = [val for val in connection]
l


#%%
Network = riskFactorNetwork()
outcomes = ['diabetes', 'stroke', 'attack', 'angina']
question2(Network, outcomes)
Network.change_edge(['income'], outcomes)
question2(Network, outcomes)



#%%
Network = riskFactorNetwork()
outcomes = ['diabetes', 'stroke', 'attack', 'angina']
question3(Network, outcomes)

#%%
