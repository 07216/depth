# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 15:18:49 2016

@author: Zhan
"""

from sklearn.decomposition import PCA

class manifoldLearner:

    def __init__(self):
        self.learner = PCA(5)
        
    def train(self,img,wSize,wX,wY):
        self.learner.fit(img)
        print sum(self.learner.__getattribute__('explained_variance_ratio_'))
        return self.learner
        
    def transer(self,obj):
        return self.learner.transform(obj)
    
    def invtranser(self,obj):
        return self.learner.inverse_transform(obj)