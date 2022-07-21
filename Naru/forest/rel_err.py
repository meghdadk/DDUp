import os
import pandas as pd


predictions = "results/"

groundtruth = "benchmark/newdata/"

rel_errors = []
for i in range(16):
    gt = pd.read_csv(groundtruth+str(i).zfill(2)+'.csv',header=None)
    es = pd.read_csv(predictions+str(i)+'.csv')
    gt = gt[0].tolist()
    es = es['est_card'].tolist()
    er = 0
    for g,e in zip(gt,es):
        er += 100*abs(g-e)/g
        
    er = er/len(gt)
    
    rel_errors.append(er)
    
print(rel_errors)
