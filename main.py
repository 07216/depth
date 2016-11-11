# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 15:01:33 2016

@author: Zhan
"""

import manifoldLearner
import imgProcesser
import tester
import calcuer
import cv2
import numpy as np
import time
import cPickle

wSize = 15
wX = 435
wY = 435
procer = imgProcesser.imgProcesser()
img = procer.pickOriginImg(wSize,wX,wY)

learner = manifoldLearner.manifoldLearner();
learner.train(img,wSize,wX,wY)


#Accuracy Test
'''
tX = 400
tY = 400
testImg,label = procer.pickTestImg(wSize,tX,tY)
lowDim = learner.transer(testImg)
tester.test(lowDim,label,wSize,tX,tY)
'''

sX=435
sY=435
width = 100
height = 100

#根据130幅图计算每个位置的系数,作为存储在rom中的数据
cal = calcuer.calcuer(procer,learner,wSize)
#根据存储的数据复原回高维，同时将实时数据也降维后复原到高维。将不同patch的数据拼凑在一起，与实时数据运算后结果对比。

f=open('save.dat','wb')
cPickle.dump(cal.code,f)

print time.clock()
depth = cal.f(sX,sY,width,height,procer.ImgLib[1])
img = cal.depImg(depth,width,height)
print time.clock()
print depth
