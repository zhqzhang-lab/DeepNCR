#!/usr/bin/python3
import numpy as np
import sys
import os
import pandas as pd
import scipy
import getopt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from decimal import Decimal, ROUND_HALF_UP

if len(sys.argv) < 2:
    print("Please input parameter files or use -h for help")
    sys.exit()

try:
    options, args = getopt.getopt(sys.argv[1:], "hc:s:p:o:", ["help", "coreset=", "score=", "prefer=", "output="])
except getopt.GetoptError:
    sys.exit()

def usage():
    print("-c or --coreset: specify the location of 'CoreSet.dat' (or a subset data file) in the CASF-2016 package")
    print("-s or --score: input your scoring file name. Remember the 1st column name is #code and the 2nd column name is score. Supported file separators are comma(,), tabs(\\t) and space character( )")
    print("-p or --prefer: input 'negative' or 'positive' string, depend on your scoring function preference")
    print("-o or --output: input the prefix of output result files. Default is My_Ranking_Power")
    print("-h or --help: print help message")
    print("\nExample: python3 ranking_power.py -c CoreSet.dat -s ./examples/X-Score.dat -p 'positive' -o 'X-Score' > MyRankingPower.out")

# Define the Predictive Index function
def cal_PI(df):
    dfsorted = df.sort_values(['logKa'], ascending=True)
    W = []
    WC = []
    lst = list(dfsorted.index)
    for i in range(5):
        xi = lst[i]
        score = float(dfsorted.loc[xi]['score'])
        bindaff = float(dfsorted.loc[xi]['logKa'])
        for j in range(i+1, 5):
            xj = lst[j]
            scoretemp = float(dfsorted.loc[xj]['score'])
            bindafftemp = float(dfsorted.loc[xj]['logKa'])
            w_ij = abs(bindaff - bindafftemp)
            W.append(w_ij)
            if score < scoretemp:
                WC.append(w_ij)
            elif score > scoretemp:
                WC.append(-w_ij)
            else:
                WC.append(0)
    pi = float(sum(WC)) / float(sum(W))
    return pi

def dec(x, y):
    if y == 2:
        return Decimal(x).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if y == 3:
        return Decimal(x).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
    if y == 4:
        return Decimal(x).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

# Read the CoreSet.dat and scoring results file    
out = 'My_Ranking_Power'
for name, value in options:
    if name in ("-h", "--help"):
        usage()
        sys.exit()
    if name in ("-c", "--coreset"):
        with open(value, 'r') as f:
            with open('cstemp', 'w+') as f1:
                for i in f.readlines():
                    if i.startswith('#'):
                        if i.startswith('#code'):
                            f1.writelines(i)
                        else:
                            continue
                    else:
                        f1.writelines(i)
        aa = pd.read_csv('cstemp', sep='[,,\t, ]+', engine='python')
        aa = aa.drop_duplicates(subset=['#code'], keep='first')
    if name in ("-s", "--score"):
        filename = value
        bb = pd.read_csv(value, sep='[,,\t, ]+', engine='python')
    if name in ("-p", "--prefer"):
        fav = value
    if name in ("-o", "--output"):
        out = value

# Process the data
testdf1 = pd.merge(aa, bb, on='#code')
if str(fav) == 'negative':
    testdf1['score'] = testdf1['score'].apply(np.negative)
    group = testdf1.groupby('target')
elif str(fav) == 'positive':
    group = testdf1.groupby('target')
else:
    print('Please input negative or positive')
    sys.exit()

# Get the representative complex in each cluster
def top(df, n=1, column='logKa'):
    return df.sort_values(by=column).iloc[-n:]

toptardf = testdf1.groupby('target').apply(top)
targetlst = toptardf['#code'].tolist()

# Calculate the Spearman correlation coefficient, Kendall correlation coefficient, and Predictive index
spearman = pd.DataFrame(index=targetlst, columns=['spearman'])
kendall = pd.DataFrame(index=targetlst, columns=['kendall'])
PI = pd.DataFrame(index=targetlst, columns=['PI'])
rankresults = pd.DataFrame(index=range(1, len(targetlst)+1), columns=['Target', 'Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5'])

tmp = 1
for i, j in group:
    testdf2 = group.get_group(i)[['#code', 'logKa', 'score']]
    testdf2 = testdf2.sort_values('score', ascending=False)
    tartemp = top(testdf2)['#code'].tolist()
    tar = ''.join(tartemp)
    if len(testdf2) == 5:
        spearman.loc[tar, 'spearman'] = testdf2.corr('spearman')['logKa']['score']
        kendall.loc[tar, 'kendall'] = testdf2.corr('kendall')['logKa']['score']
        PI.loc[tar, 'PI'] = cal_PI(df=testdf2)
        rankresults.loc[tmp, 'Rank1'] = ''.join(testdf2.iloc[0:1]['#code'].tolist())
        rankresults.loc[tmp, 'Rank2'] = ''.join(testdf2.iloc[1:2]['#code'].tolist())
        rankresults.loc[tmp, 'Rank3'] = ''.join(testdf2.iloc[2:3]['#code'].tolist())
        rankresults.loc[tmp, 'Rank4'] = ''.join(testdf2.iloc[3:4]['#code'].tolist())
        rankresults.loc[tmp, 'Rank5'] = ''.join(testdf2.iloc[4:5]['#code'].tolist())
        rankresults.loc[tmp, 'Target'] = tar
        tmp += 1
    else:
        spearman.drop(tar, inplace=True)
        kendall.drop(tar, inplace=True)
        PI.drop(tar, inplace=True)

# Print the output of ranking power evaluation
spearmanmean = dec(float(spearman['spearman'].sum()) / float(spearman.shape[0]), 3)
kendallmean = dec(float(kendall['kendall'].sum()) / float(kendall.shape[0]), 3)
PImean = dec(float(PI['PI'].sum()) / float(PI.shape[0]), 3)
tmplen = len(PI)

spearman.to_csv(f'{out}_Spearman.results', sep='\t', index_label='#Target')
kendall.to_csv(f'{out}_Kendall.results', sep='\t', index_label='#Target')
PI.to_csv(f'{out}_PI.results', sep='\t', index_label='#Target')

if os.path.exists('cstemp'):
    os.remove('cstemp')

# Output results
rankresults.dropna(axis=0, inplace=True)
rankresults.style.set_properties(align="right")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(rankresults)
print("\nSummary of the ranking power: ===========================================")
print(f"The Spearman correlation coefficient (SP) = {spearmanmean:.3f}")
print(f"The Kendall correlation coefficient (tau) = {kendallmean:.3f}")
print(f"The Predictive index (PI) = {PImean:.3f}")
