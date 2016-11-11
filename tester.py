# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 15:22:37 2016

@author: Zhan
"""

import random
import numpy as np
from scipy.spatial import KDTree

def test(data,label,wSize,tX,tY):
    n=len(data)
    randOrd = random.sample(range(0,n),n)
    randData = data[randOrd,:]
    randLabel = label[randOrd]
    #print randData
    m=5
    divid=int(n/m)
    error=[]
    for i in range(0,m):
        cpData = np.vstack([randData[0:i*divid,:],randData[(i+1)*divid:n,:]])
        cpLabel = np.concatenate((randLabel[0:i*divid],randLabel[(i+1)*divid:n]))
        testData = randData[i*divid:(i+1)*divid,:]
        testLabel = randLabel[i*divid:(i+1)*divid]        
        tree=KDTree(cpData,leafsize=2)
 #       tree = NearestNeighbors(3,algorithm='kd_tree').fit(cpData)
        p=0
        for row in testData:
            error.append(abs(cpLabel[tree.query(row)[1]]-testLabel[p]))
         #   print cpLabel[tree.kneighbors(row)[1][0]]
          #  error.append(abs(cpLabel[tree.kneighbors(row)[1]]-testLabel[p]))
            p+=1
    #print error
    error=np.array(error)>0
    print "Accuracy Rate:",100*(1-(float(sum(error)))/n),'%'    
    return 100*(1-(float(sum(error)))/n)
