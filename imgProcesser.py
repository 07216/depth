# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 15:05:17 2016

@author: Zhan
"""

import numpy as np
import cv2
import os

class imgProcesser:
    
    def __init__(self):    
        self.originImgLib = []
        self.testImgLib = [] 
        self.ImgLib = []
        self.label = []
        self.readImg()
        self.readTestImg()
        self.readTruth()    
    
    def pickOriginImg(self,wSize=13,wX=0,wY=0):
        num_img = 121
        realData = []
        for k in range(0,num_img):    
            row = []
            for i in range(wX-wSize,wX+wSize):
                for j in range(wY-wSize,wY+wSize):
                    row.append(self.originImgLib[k][i,j][0])
            realData.append(row)
        return realData
    
    def pickTestImg(self,wSize=13,wX=0,wY=0):
        num_img = 130
        realData = []
        for k in range(0,num_img):
            row = []
            for i in range(wX-wSize,wX+wSize):
                for j in range(wY-wSize,wY+wSize):
                    row.append(self.testImgLib[k][i,j][0])
            #row -= np.mean(row)
            #row /= np.sqrt(np.var(row))
            realData.append(row)
        return realData,self.label

    def pickImg(self,wSize=13,wX=0,wY=0):
        num_img = 9
        realData = []
        for k in range(0,num_img):
            row = []
            for i in range(wX-wSize,wX+wSize):
                for j in range(wY-wSize,wY+wSize):
                    row.append(self.ImgLib[k][i,j][0])
            #row -= np.mean(row)
            #row /= np.sqrt(np.var(row))
            realData.append(row)
        return realData

    
    def readImg(self):
        num_img = 121
        for i in range(1,num_img+1):
            self.originImgLib.append(cv2.imread(r"D:\stfile\Documents\cs\spatern\2016.2.29\\"+str(i)+".bmp"))
        return 1
    
    def readTestImg(self):
        p=0
        for root, dirs, files in os.walk(r"RAW2PNG"):
            files.sort(key = lambda x:int(x[4:-5]))
            for realfile in files:
                self.testImgLib.append(cv2.imread(r"RAW2PNG\\"+realfile))
                self.label.append(p/10)
                #Check Order
                #print realfile,p
                p = p+1
        self.label = np.array(self.label)
        return 1
        
    def readTruth(self):
        for root, dirs, files in os.walk(r"testImg"):
            #files.sort(key = lambda x:int(x[5:-5]))
            for realfile in files:
                self.ImgLib.append(cv2.imread(r"testImg\\"+realfile))
        return 1
        