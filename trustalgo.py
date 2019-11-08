#Run in venv
#source work/bin/activate

import os 
import numpy as np 
from sortedcontainers import SortedDict
from xlwt import Workbook 
from xlwt import easyxf
import matplotlib.pyplot as plt
import scipy.stats as st
import networkx as nx

n=int(input("\nTest nodes in MANET : "))
print("\n")
vector=[]
 
wb = Workbook() 
sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True) 
style1 = easyxf('pattern: pattern solid, fore_colour red;')
sheet1.write(0, 0, 'Manet No') 
sheet1.write(0, 1, 'PMF')
sheet1.write(0, 2, 'Trust F')  

for i in range(1,n+1):
    sheet1.write(i, 0, i) 

td = "%s - %s" % ("Test",n)
if not os.path.isdir(td):
    os.makedirs(td)


for i in range(0,n):
    temp=np.random.randint(1,9,3)
    #4 Elements in Vector, Mean of Temperature,Humidity and Light.
    vector.append(temp)

cosinedictionary={}

def bern_post(n_params=200, n_sample=200, true_p=.8, prior_p=.5, n_prior=n):
    params = np.linspace(0, 1, n_params)
    sample = np.random.binomial(n=1, p=true_p, size=n_sample)
    likelihood = np.array([np.product(st.bernoulli.pmf(sample, p)) for p in params])
    prior_sample = np.random.binomial(n=1, p=prior_p, size=n_prior)
    prior = np.array([np.product(st.bernoulli.pmf(prior_sample, p)) for p in params])
    prior = prior / np.sum(prior)
    posterior = [prior[i] * likelihood[i] for i in range(prior.shape[0])]
    posterior = posterior / np.sum(posterior)
    return np.mean(posterior)

for i in range(0,n+1):
    for j in range(i+i,n):
        res=np.dot(vector[i],vector[j])
        k = bern_post(res,res,0.8,.5,n)
        sheet1.write(j+1, 1, k) 
        cosinedictionary[k]=[i,j]

numberofuntrystablenodes = np.random.randint(1,20)
trustfactor={}

nodecombinationdict={}
def remove_duplicatenodecombination():
    for key,value in cosinedictionary.items():
        if value not in nodecombinationdict.values():
            nodecombinationdict[key]=value

remove_duplicatenodecombination()

def calculate_similarity():
    for i in range(0,n):
        sum=0
        for key,value in nodecombinationdict.items():
            if int(value[0])==i or int(value[1]==i):
                sum=sum+key
        trustfactor[sum]=i

calculate_similarity()
result=SortedDict(trustfactor)

factorsum=0
for key,value in result.items():
    factorsum += key

trustvaluedictres={}

for key,value in result.items():
    trustvaluedictres[key/factorsum]=value

t_n = []
hh = 0
print("\nList of Calculated Trust Factors:\n")
for i in trustvaluedictres:
    print(trustvaluedictres[i],i)
    sheet1.write(trustvaluedictres[i] + 1, 2, i) 


c=0
g1 = [[],[]]
g2 = [[],[]]

#Untrustworthy Node List 
print("\nList of all Untrustable nodes:\n")
for key,value in trustvaluedictres.items():
    if c!=numberofuntrystablenodes and key < 0.04:
        print("\nUntrustable node :",value + 1," \nTrustfactor :",key)
        c += 1
        sheet1.write(int(value) + 1, 2, key,style1) 
        t_n.append(int(value) + 1)
        g2[0].append(value)
        g2[1].append(key)
    else:
        g1[0].append(value)
        g1[1].append(key)

sorted(t_n)

#Creating Scatter 
def scplot():
    data = (g1, g2)
    colors = ("green", "red")
    groups = ("Untrustable", "Trustable")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Scatter Plot of Nodes')
    plt.legend(loc=2)
    save_results_to = '/Users/anurag/Desktop/IOTTE/' + td + '/'
    plt.savefig(save_results_to + 'scplot.png')
    plt.show()

t_t =[]
for i in range(1,n+1):
    if i not in t_n:
        t_t.append(i)    

#Creating Network
def digr():
    G_symmetric = nx.Graph()
    for i in t_t:
        for j in t_t:
            if(i!=j):
                G_symmetric.add_edge(i,j)
    for i in t_n:
        for j in t_n:
            if(i==j):
                G_symmetric.add_edge(i,j)
    nx.draw_networkx(G_symmetric,pos=nx.spring_layout(G_symmetric,k=0.5,iterations=20),node_size=60,font_size=8)
    save_results_to = '/Users/anurag/Desktop/IOTTE/' + td + '/'
    plt.savefig(save_results_to + 'finalNet.png')
    plt.show()


scplot()
digr()  
wb.save(td + '/%s.xls' % (td)) 
