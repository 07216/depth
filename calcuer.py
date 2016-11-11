# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 17:43:23 2016

@author: Zhan
"""

import numpy as np
import cPickle
from scipy.spatial import KDTree

class calcuer:

    def __init__(self,procer,learner,wSize):
        self.code = np.zeros((50,50,20,5))
        self.Img = []
        self.learner = learner
        self.procer = procer
        self.wSize = wSize
        self.encoding()
        
    def depImg(depth,width,height):
        offset = 0
        Focal = 1118.5
        L = 190.66
        Img = np.zeros(width,height,dtype=np.int)
        for i in range(0,height):
            for j in range(0,width):
                Img[i,j] = 1/(1.0/(1+0.1*depth[i,j])-offset/Focal/L)
        return Img[i,j]
        
#获得降维后存储的数据
    def encoding(self):
        sX = self.wSize
        sY = self.wSize
        i = 0
        while(sX<960-self.wSize):
            j = 0
            sY = self.wSize
            while(sY<1280-self.wSize):
                tmpImg,tmplabel = self.procer.pickTestImg(self.wSize,sX,sY)
                tmpImg = np.array(tmpImg)
                for k in range(0,13):
                    tmp = self.learner.transer(tmpImg[k,:])[0]
                    for l in range(0,5):
                        self.code[i,j,k,l] = tmp[l]
                j = j+1
                sY = sY + 2*self.wSize
            i = i+1
            sX = sX + 2*self.wSize
        return 1

    def codeFromFile(self,f):
        cPickle.dump(self.code,f)
    
    def depth(self,sX,sY):
        i = (sX-self.wSize)/(2*self.wSize)
        j = (sY-self.wSize)/(2*self.wSize)
        tmp = self.code[i,j,0:13,:]
        print tmp
        tree=KDTree(tmp,leafsize=2)
        return tree.query(self.learner.transer(self.cut(sX,sY)))[1]        
        
    def cut(self,sX,sY):
        realData = []
        for i in range(sX-self.wSize,sX+self.wSize):
            for j in range(sY-self.wSize,sY+self.wSize):
                realData.append(self.Img[i,j][0])
        return realData

    def decom(self,i,j,d):
        de = self.learner.invtranser(self.code[i,j,d])
        w = self.wSize
        res = np.zeros((2*w,2*w))
        p = 0
        for i in range(0,2*w):
            for j in range(0,2*w):
                res[i,j] = de[p]
                p = p+1
        return res

#核心部分，返回深度值为d的图像中，与sX,sY相交的几块图像。返回为图像的数组，以及解释图像位置的flag与图像相交情况的flag
    def recover(self,sX,sY,d):
        result = []
        imgRec = []
        w = self.wSize
        dw = 2*self.wSize
        if (sX-w) % dw ==0 and (sY-w)%dw == 0:
            result.append([1])
            imgRec.append(self.decom((sX-w)/dw,(sY-w)/dw,d))
        elif (sX-w) % dw ==0:
            result.append([2])
            imgRec.append(self.decom((sX-w)/dw,(sY-w)/dw,d))
            imgRec.append(self.decom((sX-w)/dw,(sY-w)/dw+1,d))
        elif (sY-w) % dw ==0:
            result.append([3])
            imgRec.append(self.decom((sX-w)/dw,(sY-w)/dw,d))
            imgRec.append(self.decom((sX-w)/dw+1,(sY-w)/dw,d))
        else:
            result.append([4])
            imgRec.append(self.decom((sX-w)/dw,(sY-w)/dw,d))
            imgRec.append(self.decom((sX-w)/dw,(sY-w)/dw+1,d))
            imgRec.append(self.decom((sX-w)/dw+1,(sY-w)/dw+1,d))
            imgRec.append(self.decom((sX-w)/dw+1,(sY-w)/dw,d))
        result.append(imgRec)
        return result
            
#合成图片
    def gather(self,current,sX,sY):
        w = self.wSize
        dw = 2*self.wSize
        resImg = np.zeros((dw,dw))
        if current[0][0] == 1:
            return current[1][0]
        elif current[0][0] == 2:
            resImg[:,0:(dw-(sY-w)%dw)]=current[1][0][:,(sY-w)%dw:dw]
            resImg[:,dw-(sY-w)%dw:dw]=current[1][1][:,0:(sY-w)%dw]
        elif current[0][0] == 3:
            resImg[0:(dw-(sX-w)%dw),:]=current[1][0][(sX-w)%dw:dw,:]
            resImg[(dw-(sX-w)%dw):dw,:]=current[1][1][0:(sX-w)%dw,:]
        else:
            resImg[0:(dw-(sX-w)%dw),0:(dw-(sY-w)%dw)]=current[1][0][(sX-w)%dw:dw,(sY-w)%dw:dw]
            resImg[0:(dw-(sX-w)%dw),(dw-(sY-w)%dw):dw]=current[1][1][(sX-w)%dw:dw,0:(sY-w)%dw]
            resImg[(dw-(sX-w)%dw):dw,(dw-(sY-w)%dw):dw]=current[1][2][0:(sX-w)%dw,0:(sY-w)%dw]
            resImg[(dw-(sX-w)%dw):dw,0:(dw-(sY-w)%dw)]=current[1][3][(dw-(sX-w)%dw):dw,(sY-w)%dw:dw]
        return resImg
        
    def vectorize(self,r):
        result = []
        for i in range(0,2*self.wSize):
            for j in range(0,2*self.wSize):
                result.append(r[i,j])
        return result

#在tmp提供的3幅中（可能由于深度在上下限仅有两幅）找出与real最接近的一副，注意real已经拉成了向量        
    def comp(self,tmp,real):
        final = self.learner.transer(real)[0]
        Lp = 999999999.
        p = 0
        for i in range(0,3):
            if tmp[i] == []:
                continue
            vecTemp = self.vectorize(tmp[i]) 
            rec = self.learner.transer(vecTemp)[0]
            rec = rec - final
            if(np.dot(rec,rec)<Lp):
                Lp = np.dot(rec,rec)
                p = i
        if p==0 :
            return 0
        elif p==1:
            return 1
        else:
            return -1
        
#计算深度图
    def f(self,sX,sY,width,height,Img):
#图片存档反复使用
#以(sX,sY)为起点，宽width,长height的一个区域的深度图
        self.Img = Img
        
        result = np.zeros((height,width))
#在sX,sY为中心的patch处利用KDTree做KNN准确找到深度图，并得到该点深度信息
        d = self.depth(sX,sY)
        result[0,0] = d
        current = []

        w = self.wSize
        dw = 2* self.wSize        
        
        for i in range(0,height):
#滚动更新，在更新i,j时检查是否需要更新
            if i!=0:
                d = result[i-1,0]
                current = []
                current.append(self.recover(sX+i,sY,d))
                if d < 12:
                    current.append(self.recover(sX+i,sY,d+1))
                else:
                    current.append([])
                if d > 0:
                    current.append(self.recover(sX+i,sY,d-1))
                else:
                    current.append([])
            for j in range(0,width):
                if i!=0 and j!=0:
                    if j%dw == 1:                                        
                        d = result[i,j-1]
                        current = []
                        current.append(self.recover(sX+i,sY+j,d))
                        if d < 12:
                            current.append(self.recover(sX+i,sY+j,d+1))
                        else:
                            current.append([])
                        if d > 0:
                            current.append(self.recover(sX+i,sY+j,d-1))
                        else:
                            current.append([])
#根据重构结果拼合成图，再做PCA，最后对比合成图PCA结果与实际图片的结果，从3个深度中选择一个深度
                    tmp = []
                    for k in range(0,3):
                        if current[k] == []:
                            tmp.append([])
                        else:
                            tmp.append(self.gather(current[k],sX+i,sY+j))
                    d = d + self.comp(tmp,self.cut(sX+i,sY+j))
                    result[i,j] = d
        return result